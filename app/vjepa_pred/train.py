# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import copy
import gc
import random
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from app.vjepa.transforms import make_transforms
from app.vjepa.utils import init_opt, init_video_model
from src.datasets.data_manager import init_data
from src.masks.multiseq_multiblock3d import MaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer
from src.utils.checkpoint_loader import robust_checkpoint_loader
import torch.distributed as dist
from app.vjepa_pred.losses import JepaLoss, MotionLoss
# --
log_timings = True
log_freq = 10
CHECKPOINT_FREQ = 1
GARBAGE_COLLECT_ITR_FREQ = 50
# --

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


logger = get_logger(__name__, force=True)


from scipy.optimize import linear_sum_assignment

class MotionHead(torch.nn.Module):
    """
    Simple Dense Predictor for Motion Intensity.
    Enhanced with 3 layers and LayerNorm for better stability.
    Input: [B, N, D]
    Output: [B, N] in [0, 1]
    """
    def __init__(self, embed_dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim
            
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, N_pred, D]
        return self.mlp(x).squeeze(-1)


class MultiLayerProjector(torch.nn.Module):
    """
    Projector for Hierarchical Feature Alignment.
    Maps predictor output to multiple target layers.
    """
    def __init__(self, in_dim, out_dim, target_layer_ids):
        super().__init__()
        self.target_layer_ids = target_layer_ids
        # Last layer usually aligns directly, others need projection
        self.heads = torch.nn.ModuleList([
            torch.nn.Linear(in_dim, out_dim) if i != target_layer_ids[-1] else torch.nn.Identity()
            for i in target_layer_ids
        ])

    def forward(self, z):
        # z: [B, N, D]
        # Return list of projected features corresponding to target_layer_ids
        return [head(z) for head in self.heads]


def get_motion_target(clips, patch_size, tubelet_size, sigma=1.0, threshold=0.2):
    """
    计算 Patch 级别的运动指标，并应用高斯平滑以获得更稳健的分布 Target。
    """
    if isinstance(clips, list):
        return [get_motion_target(c, patch_size, tubelet_size, sigma, threshold) for c in clips]

    B, C, T, H, W = clips.shape
    
    # 1. 时间梯度 (I_t): 帧差 [B, C, T, H, W]
    diffs = torch.abs(clips[:, :, 1:] - clips[:, :, :-1])
    diffs = torch.cat([diffs[:, :, :1], diffs], dim=2) # 补齐时间维
    
    # 2. 空间梯度 (I_x, I_y): 纹理强度 [B, C, T, H, W]
    # 计算 x 方向梯度: I(x+1) - I(x)
    dx = clips[:, :, :, :, 1:] - clips[:, :, :, :, :-1]
    dx = torch.cat([dx, dx[:, :, :, :, -1:]], dim=4) # Padding
    
    # 计算 y 方向梯度: I(y+1) - I(y)
    dy = clips[:, :, :, 1:, :] - clips[:, :, :, :-1, :]
    dy = torch.cat([dy, dy[:, :, :, -1:, :]], dim=3) # Padding
    
    # 空间梯度模长 (Texture Strength)
    # 加上一个小常数防止开根号梯度消失
    spatial_grad = torch.sqrt(dx**2 + dy**2 + 1e-6)
    
    # 3. 计算物理速度 (Speed)
    alpha = 0.05 
    speed_map = diffs / (spatial_grad + alpha)
    
    # 过滤低纹理区域 (Low Texture Filtering)
    # 在纹理极弱的区域(如纯黑背景、反光过曝)，空间梯度接近0，会导致计算出的速度极不稳定(噪点)。
    # 我们将这些区域的速度置零，避免脏数据干扰。
    valid_texture = spatial_grad > 0.01
    speed_map = speed_map * valid_texture.float()
    
    # --- 关键步骤 3.1: Global Motion Estimation & Removal ---
    # 手术场景假设：
    # 1. 背景(Global Motion)占据大部分面积，且运动一致性较高。
    # 2. 器械(Local Motion)占据小部分面积，且运动强度通常显著高于背景。
    # 我们使用 Median 来稳健估计 Global Motion (相机运动)。
    # speed_map: [B, C, T, H, W]
    
    # 计算每帧的全局运动基准 (Median over H, W)
    flat_speed = speed_map.view(B, C, T, -1) # [B, C, T, H*W]
    global_motion_est = torch.quantile(flat_speed, 0.5, dim=-1, keepdim=True) # [B, C, T, 1]
    global_motion_est = global_motion_est.view(B, C, T, 1, 1)
    
    # --- 策略调整：分级幅度控制 (Hierarchical Amplitude) ---
    # 目的：防止剧烈的相机运动(Global)导致全图饱和(变为全1)，掩盖了器械(Local)
    # 方案：
    # 1. Global Base: 限制最大贡献度 (例如 max 0.3)
    # 2. Local Motion: 允许贡献剩余的幅度 (0.7)
    
    # Global Base Compression:
    # 使用 tanh 压缩，但乘以较小的系数，并强制截断
    # 这里的 5.0 是灵敏度，0.3 是最大底噪亮度 (灰度)
    global_base_map = torch.tanh(global_motion_est * 5.0).clamp(max=0.3)
    global_base_map = global_base_map.expand_as(speed_map)
    
    # Local Motion Compression:
    # 我们不再硬性扣除，而是将其分离并作为平缓的底噪叠加回来
    # Local Motion: 显著高于背景的部分
    # 计算 local_motion_map 时，我们希望它是 relative signal
    local_motion_map = torch.relu(speed_map - (global_motion_est * 1.2))
    
    # 对 Local 进行归一化，使其占据 0.0 ~ 0.7 的动态范围
    # 注意：Local Motion 是极其稀疏的，我们希望显著的峰值接近 0.7
    local_motion_map = torch.tanh(local_motion_map * 5.0) * 0.7
    
    # 合成 Target: Target = Base(max 0.3) + Local(max 0.7) -> Total max 1.0
    # 即使相机剧烈运动，global_base_map 也止步于 0.3，器械依然能叠加出更亮的值
    combined_map = local_motion_map + global_base_map
    
    # 使用 combined_map 进行后续的平均
    combined_map = combined_map.mean(dim=1, keepdim=True) # [B, 1, T, H, W]

    # --- 关键步骤 3.5: 局部阈值去噪 (仅针对 Local 部分) ---
    # 这里不需要额外的阈值处理了，因为 relu(speed - 1.2*global) 已经起到了动态阈值的作用
    
    # 4. 降采样到 Patch Grid
    t_grid, h_grid, w_grid = T // tubelet_size, H // patch_size, W // patch_size
    motion_target = F.adaptive_avg_pool3d(
        combined_map, 
        output_size=(t_grid, h_grid, w_grid)
    ) # [B, 1, t_grid, h_grid, w_grid]
    
    # 5. --- 关键优化：高斯平滑 (Gaussian Smoothing) ---
    # 这将独立的离散点平滑为混合高斯分布 (Mixture of Gaussians)
    if sigma > 0:
        # Dynamic kernel size based on sigma
        k_size = int(4 * sigma + 1)
        if k_size % 2 == 0:
            k_size += 1

        # Create Gaussian kernel
        x = torch.arange(k_size, device=clips.device, dtype=clips.dtype) - (k_size - 1) / 2
        k = torch.exp(-0.5 * (x / sigma)**2)
        k = k / k.sum()

        k_3d = k[:, None, None] * k[None, :, None] * k[None, None, :]
        k_3d = k_3d[None, None, ...] # [1, 1, k, k, k]
        
        # Padding 保持尺寸不变
        padding = k_size // 2
        motion_target = F.conv3d(motion_target, k_3d, padding=padding)

    motion_target_flat = motion_target.flatten(1) # [B, N_tokens]
    
    # 6. 归一化到 [0, 1]
    # 前面的逻辑已经显式控制了数值范围：Base(max 0.3) + Local(max 0.7) -> Max 1.0
    # 因此这里不再需要激进的 tanh 压缩，直接截断保证数值安全即可。
    # 这样能保留 0.3(背景) 和 >0.3(前景) 之间的线性差异，让模型更容易区分。
    motion_target = motion_target_flat.clamp(0.0, 1.0)
    
    return motion_target.detach()


def load_cpt_checkpoint(encoder, pretrained, checkpoint_key="target_encoder"):
    logger.info(f"Loading pretrained model from {pretrained}")
    checkpoint = robust_checkpoint_loader(pretrained, map_location="cpu")
    pretrained_dict = checkpoint[checkpoint_key]
    
    
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    # pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f"key '{k}' could not be found in loaded state dict")
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f"{pretrained_dict[k].shape} | {v.shape}")
            logger.info(f"key '{k}' is of different shape in model and loaded state dict")
            exit(1)
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    # print(encoder)
    logger.info(f"loaded pretrained {checkpoint_key} with msg: {msg}")
    logger.info(f"loaded pretrained {checkpoint_key} from epoch: {checkpoint['epoch']}\n path: {pretrained}")
    del checkpoint
    return encoder


def load_checkpoint_with_module_fix(
    r_path,
    encoder,
    predictor,
    target_encoder,
    opt,
    scaler,
    motion_head=None,
):
    """加载checkpoint并处理module.前缀不一致的问题"""
    logger.info(f"Loading checkpoint from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
    
    epoch = checkpoint["epoch"]
    
    def fix_state_dict_keys(checkpoint_dict, model_state_dict):
        """智能处理module.前缀：检查模型是否需要module.前缀，然后调整checkpoint中的键"""
        # 检查模型的键是否有module.前缀
        model_keys = list(model_state_dict.keys())
        has_module = any(k.startswith("module.") for k in model_keys)
        
        # 检查checkpoint的键是否有module.前缀
        ckpt_keys = list(checkpoint_dict.keys())
        ckpt_has_module = any(k.startswith("module.") for k in ckpt_keys)
        
        # 如果模型有module.前缀但checkpoint没有，添加前缀
        if has_module and not ckpt_has_module:
            fixed_dict = {f"module.{k}": v for k, v in checkpoint_dict.items()}
            logger.info(f"Adding 'module.' prefix to checkpoint keys (model has module, checkpoint doesn't)")
        # 如果模型没有module.前缀但checkpoint有，去除前缀
        elif not has_module and ckpt_has_module:
            fixed_dict = {k.replace("module.", ""): v for k, v in checkpoint_dict.items()}
            logger.info(f"Removing 'module.' prefix from checkpoint keys (checkpoint has module, model doesn't)")
        else:
            # 两者一致，直接使用
            fixed_dict = checkpoint_dict
        
        return fixed_dict
    
    # -- loading encoder，处理"module."前缀
    pretrained_dict = checkpoint["encoder"]
    pretrained_dict = fix_state_dict_keys(pretrained_dict, encoder.state_dict())
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")
    
    # -- loading predictor，处理"module."前缀
    pretrained_dict = checkpoint["predictor"]
    pretrained_dict = fix_state_dict_keys(pretrained_dict, predictor.state_dict())
    msg = predictor.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"loaded pretrained predictor from epoch {epoch} with msg: {msg}")
    
    # -- loading target_encoder，处理"module."前缀
    if target_encoder is not None:
        pretrained_dict = checkpoint["target_encoder"]
        pretrained_dict = fix_state_dict_keys(pretrained_dict, target_encoder.state_dict())
        msg = target_encoder.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained target encoder from epoch {epoch} with msg: {msg}")
    
    # -- loading optimizer
    opt.load_state_dict(checkpoint["opt"])
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    logger.info(f"loaded optimizers from epoch {epoch}")
    
    # -- loading motion_head
    if motion_head is not None:
        if "motion_head" in checkpoint:
            pretrained_dict = checkpoint["motion_head"]
            pretrained_dict = fix_state_dict_keys(pretrained_dict, motion_head.state_dict())
            msg = motion_head.load_state_dict(pretrained_dict, strict=False)
            logger.info(f"loaded pretrained motion_head from epoch {epoch} with msg: {msg}")
        else:
            logger.info("checkpoint does not contain motion_head, skipping load")

    logger.info(f"read-path: {r_path}")
    del checkpoint
    
    return (
        encoder,
        predictor,
        target_encoder,
        opt,
        scaler,
        epoch,
        motion_head,
    )


def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    folder = args.get("folder")
    cfgs_meta = args.get("meta")
    load_model = cfgs_meta.get("load_checkpoint") or resume_preempt
    r_file = cfgs_meta.get("read_checkpoint", None)
    seed = cfgs_meta.get("seed", _GLOBAL_SEED)
    save_every_freq = cfgs_meta.get("save_every_freq", -1)
    skip_batches = cfgs_meta.get("skip_batches", -1)
    use_sdpa = cfgs_meta.get("use_sdpa", False)
    sync_gc = cfgs_meta.get("sync_gc", False)
    which_dtype = cfgs_meta.get("dtype")
    logger.info(f"{which_dtype=}")
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    # -- MASK
    cfgs_mask = args.get("mask")

    # -- MODEL
    cfgs_model = args.get("model")
    compile_model = cfgs_model.get("compile_model", False)
    use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)
    model_name = cfgs_model.get("model_name")
    pred_depth = cfgs_model.get("pred_depth")
    pred_num_heads = cfgs_model.get("pred_num_heads", None)
    pred_embed_dim = cfgs_model.get("pred_embed_dim")
    uniform_power = cfgs_model.get("uniform_power", False)
    use_mask_tokens = cfgs_model.get("use_mask_tokens", False)
    zero_init_mask_tokens = cfgs_model.get("zero_init_mask_tokens", True)
    use_rope = cfgs_model.get("use_rope", False)
    use_silu = cfgs_model.get("use_silu", False)
    use_pred_silu = cfgs_model.get("use_pred_silu", False)
    wide_silu = cfgs_model.get("wide_silu", True)
    cpt_ckeckpoint = cfgs_model.get("cpt_ckeckpoint", None)

    # -- DATA
    cfgs_data = args.get("data")
    dataset_type = cfgs_data.get("dataset_type", "videodataset")
    dataset_paths = cfgs_data.get("datasets", [])
    datasets_weights = cfgs_data.get("datasets_weights")
    dataset_fpcs = cfgs_data.get("dataset_fpcs")
    max_num_frames = max(dataset_fpcs)
    if datasets_weights is not None:
        assert len(datasets_weights) == len(dataset_paths), "Must have one sampling weight specified for each dataset"
    batch_size = cfgs_data.get("batch_size")
    tubelet_size = cfgs_data.get("tubelet_size")
    fps = cfgs_data.get("fps")
    crop_size = cfgs_data.get("crop_size", 224)
    patch_size = cfgs_data.get("patch_size")
    pin_mem = cfgs_data.get("pin_mem", False)
    num_workers = cfgs_data.get("num_workers", 1)
    persistent_workers = cfgs_data.get("persistent_workers", True)

    # -- DATA AUGS
    cfgs_data_aug = args.get("data_aug")
    ar_range = cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3])
    rr_scale = cfgs_data_aug.get("random_resize_scale", [0.3, 1.0])
    motion_shift = cfgs_data_aug.get("motion_shift", False)
    reprob = cfgs_data_aug.get("reprob", 0.0)
    use_aa = cfgs_data_aug.get("auto_augment", False)

    # -- LOSS
    cfgs_loss = args.get("loss", {})
    motion_loss_weight = float(cfgs_loss.get("motion_loss_weight", 0.1))
    
    jepa_loss_cfg = cfgs_loss.get("jepa_loss", {})
    # Rename 'type' to 'metric' for clarity, fallback to 'type' for backward compatibility
    jepa_metric_type = jepa_loss_cfg.get("metric", jepa_loss_cfg.get("type", "lp"))
    # New location for jepa_metric_p (inside jepa_loss)
    # Fallback to top-level if not found (for backward compatibility)
    jepa_metric_p = float(jepa_loss_cfg.get("metric_p", cfgs_loss.get("jepa_metric_p", 1.0)))
    jepa_loss_params = jepa_loss_cfg.get("params") or {}
    
    # New location for multiscale config (inside jepa_loss)
    # Fallback to top-level loss.multiscale if not found
    multiscale_cfg = jepa_loss_cfg.get("multiscale", cfgs_loss.get("multiscale", {}))
    
    motion_loss_cfg = cfgs_loss.get("motion_loss", {})
    # 默认使用 Smooth L1
    # Rename 'type' to 'metric', fallback to 'type'
    motion_metric_type = motion_loss_cfg.get("metric", motion_loss_cfg.get("type", "smooth_l1")).lower()
    motion_loss_params = motion_loss_cfg.get("params") or {}
    motion_heatmap_cfg = motion_loss_cfg.get("distribution", {})
    # 默认开启 Heatmap 模式
    motion_heatmap_enabled = bool(motion_heatmap_cfg.get("enable", True))
    # Pred 保持 Logits (为了数值稳定性，在 Loss 里做 LogSoftmax)
    motion_heatmap_pred_method = motion_heatmap_cfg.get("pred_method", "none").lower()
    # Target 使用 L1 归一化 (Intensity -> Probability)
    motion_heatmap_target_method = motion_heatmap_cfg.get("target_method", "l1").lower()
    motion_heatmap_temperature = float(motion_heatmap_cfg.get("temperature", 1.0)) 
    motion_heatmap_eps = float(motion_heatmap_cfg.get("eps", 1e-6))
    motion_heatmap_topk = int(motion_heatmap_cfg.get("topk", 0))
    motion_heatmap_topk_mode = motion_heatmap_cfg.get("topk_mode", "abs").lower()
    motion_heatmap_sigma = float(motion_heatmap_cfg.get("sigma", 1.0))
    jepa_metric_type = jepa_metric_type.lower()
    motion_metric_type = motion_metric_type.lower()

    # --- New Configs for Innovation ---
    # 1. Multiscale Feature Reconstruction
    # Config format: loss.jepa_loss.multiscale: {enable: True, layers: [0.5, 1.0], weights: [0.5, 1.0]}
    # (Already parsed above into multiscale_cfg)
    use_multiscale = bool(multiscale_cfg.get("enable", False))
    # Default to last layer only if not enabled or not specified
    multiscale_layers_ratio = multiscale_cfg.get("layers", [1.0])
    multiscale_loss_weights = multiscale_cfg.get("weights", [1.0])
    assert len(multiscale_layers_ratio) == len(multiscale_loss_weights), "Layers and weights must have same length"

    # 2. Motion Aware JEPA Weighting
    # Config format: loss.motion_loss: {enable_weighted_jepa: True, weight_factor: 2.0}
    use_motion_weighted_jepa = bool(motion_loss_cfg.get("enable_weighted_jepa", False))
    motion_weight_factor = float(motion_loss_cfg.get("weight_factor", 2.0))
    # ----------------------------------

    # --- Initialize Loss Functions ---
    # Ensure defaults from jepa_metric_p are propagated
    if "p" not in jepa_loss_params:
        jepa_loss_params["p"] = jepa_metric_p
    if "power" not in jepa_loss_params:
        jepa_loss_params["power"] = jepa_metric_p

    jepa_loss_fn = JepaLoss(
        metric_type=jepa_metric_type,
        loss_params=jepa_loss_params,
        motion_weight_factor=motion_weight_factor if use_motion_weighted_jepa else 0.0
    )

    # Construct heatmap config with parsed values
    motion_heatmap_full_cfg = {
        "enable": motion_heatmap_enabled,
        "pred_method": motion_heatmap_pred_method,
        "target_method": motion_heatmap_target_method,
        "temperature": motion_heatmap_temperature,
        "eps": motion_heatmap_eps,
        "topk": motion_heatmap_topk,
        "topk_mode": motion_heatmap_topk_mode,
        "sigma": motion_heatmap_sigma
    }

    motion_loss_fn = MotionLoss(
        metric_type=motion_metric_type,
        loss_params=motion_loss_params,
        heatmap_cfg=motion_heatmap_full_cfg
    )
    # ---------------------------------


    # -- OPTIMIZATION
    cfgs_opt = args.get("optimization")
    ipe = cfgs_opt.get("ipe", None)
    ipe_scale = cfgs_opt.get("ipe_scale", 1.0)
    wd = float(cfgs_opt.get("weight_decay"))
    final_wd = float(cfgs_opt.get("final_weight_decay"))
    num_epochs = cfgs_opt.get("epochs")
    warmup = cfgs_opt.get("warmup")
    start_lr = cfgs_opt.get("start_lr")
    lr = cfgs_opt.get("lr")
    final_lr = cfgs_opt.get("final_lr")
    ema = cfgs_opt.get("ema")
    betas = cfgs_opt.get("betas", (0.9, 0.999))
    eps = cfgs_opt.get("eps", 1.0e-8)
    # ----------------------------------------------------------------------- #

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    # -- Initialize wandb (only on rank 0)
    wandb_initialized = False
    if WANDB_AVAILABLE:
        cfgs_wandb = args.get("wandb", None)
        if cfgs_wandb is not None and rank == 0:
            wandb_project = cfgs_wandb.get("project", "nsjepa-training")
            wandb_entity = cfgs_wandb.get("entity", None)
            wandb_name = cfgs_wandb.get("name", None)
            wandb_tags = cfgs_wandb.get("tags", [])
            wandb_group = cfgs_wandb.get("group", None)
            wandb_notes = cfgs_wandb.get("notes", None)
            
            # Prepare config dict for wandb
            wandb_run_config = {
                "model_name": model_name,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "lr": lr,
                "start_lr": start_lr,
                "final_lr": final_lr,
                "weight_decay": wd,
                "final_weight_decay": final_wd,
                "jepa_metric_p": jepa_metric_p,
                "motion_loss_weight": motion_loss_weight,
                "jepa_metric_type": jepa_metric_type,
                "motion_metric_type": motion_metric_type,
                "warmup": warmup,
                "crop_size": crop_size,
                "patch_size": patch_size,
                "tubelet_size": tubelet_size,
                "max_num_frames": max_num_frames,
                "ipe": ipe,
                "ipe_scale": ipe_scale,
                "dtype": which_dtype,
                "seed": seed,
                "folder": folder,
            }
            
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_name,
                config=wandb_run_config,
                tags=wandb_tags,
                group=wandb_group,
                notes=wandb_notes,
                resume="allow"
            )
            wandb_initialized = True
            logger.info(f"wandb initialized: {wandb.run.url}")
        elif cfgs_wandb is not None and rank != 0:
            logger.info("wandb will be initialized only on rank 0")
    elif args.get("wandb") is not None:
        logger.warning("wandb configuration found but wandb package is not installed. Install with: pip install wandb")

    # -- set device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_file = "latest.pt"
    latest_path = os.path.join(folder, latest_file)
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path
        if not os.path.exists(load_path):
            load_path = None
            load_model = False

    # -- make csv_logger
    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%d", "iter-time(ms)"),
        ("%d", "gpu-time(ms)"),
        ("%d", "dataload-time(ms)"),
    )

    # -- init model
    encoder, predictor = init_video_model(
        uniform_power=uniform_power,
        use_mask_tokens=use_mask_tokens,
        num_mask_tokens=10,
        zero_init_mask_tokens=zero_init_mask_tokens,
        device=device,
        patch_size=patch_size,
        max_num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_num_heads=pred_num_heads,
        pred_embed_dim=pred_embed_dim,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        use_pred_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_rope=use_rope,
        use_activation_checkpointing=use_activation_checkpointing,
    )
    
    target_encoder = copy.deepcopy(encoder)
    
    ## load pretrained
    if cpt_ckeckpoint is not None:
        # import pdb; pdb.set_trace()
        encoder = load_cpt_checkpoint(encoder, cpt_ckeckpoint, checkpoint_key="encoder")
        predictor = load_cpt_checkpoint(predictor, cpt_ckeckpoint, checkpoint_key="predictor")
        target_encoder = load_cpt_checkpoint(target_encoder, cpt_ckeckpoint, checkpoint_key="target_encoder")
        
    
    if compile_model:
        logger.info("Compiling encoder, target_encoder, and predictor.")
        torch._dynamo.config.optimize_ddp = False
        encoder.compile()
        target_encoder.compile()
        predictor.compile()

    # Get strategy selection configuration from args
    # Allow strategy_selection and strategy_weights to be specified at the mask level or in args
    strategy_selection = args.get("mask_strategy_selection", "random")  # Default: randomly select one strategy per batch
    strategy_weights = args.get("mask_strategy_weights", None)  # Optional: weights for each strategy
    
    mask_collator = MaskCollator(
        cfgs_mask=cfgs_mask,
        dataset_fpcs=dataset_fpcs,
        crop_size=crop_size,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        strategy_selection=strategy_selection,
        strategy_weights=strategy_weights,
    )
    transform = make_transforms(
        random_horizontal_flip=True,
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        reprob=reprob,
        auto_augment=use_aa,
        motion_shift=motion_shift,
        crop_size=crop_size,
    )

    # -- init data-loaders/samplers
    (unsupervised_loader, unsupervised_sampler) = init_data(
        data=dataset_type,
        root_path=dataset_paths,
        batch_size=batch_size,
        training=True,
        dataset_fpcs=dataset_fpcs,
        fps=fps,
        transform=transform,
        rank=rank,
        world_size=world_size,
        datasets_weights=datasets_weights,
        persistent_workers=persistent_workers,
        collator=mask_collator,
        num_workers=num_workers,
        pin_mem=pin_mem,
        log_dir=None,
    )
    try:
        _dlen = len(unsupervised_loader)
    except Exception:  # Different interface for webdataset
        _dlen = unsupervised_loader.num_batches
    if ipe is None:
        ipe = _dlen
    logger.info(f"iterations per epoch/dataset length: {ipe}/{_dlen}")

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        mixed_precision=mixed_precision,
        betas=betas,
        eps=eps,
    )

    # --- 新增：初始化 Motion Head ---
    # 获取 predictor 的输出维度
    # predictor 的最终输出维度被投影回 encoder 的 embed_dim
    # encoder 此时是 MultiSeqWrapper，通过 backbone 访问
    pred_dim = encoder.backbone.embed_dim
    # Reverted to simple dense MotionHead with bottleneck hidden_dim
    motion_head = MotionHead(pred_dim, hidden_dim=pred_dim // 2).to(device)

    # --- Hierarchical Feature Alignment ---
    if hasattr(encoder, "backbone"):
        enc_depth = encoder.backbone.get_num_layers()
    else:
        enc_depth = encoder.module.backbone.get_num_layers()
        
    # Determine target layers based on config
    if use_multiscale:
        target_layer_ids = []
        for r in multiscale_layers_ratio:
            # r is 0.0 ~ 1.0 (relative depth)
            idx = int(r * enc_depth) - 1
            idx = max(0, min(idx, enc_depth - 1))
            target_layer_ids.append(idx)
        # Ensure unique and sorted
        target_layer_ids = sorted(list(set(target_layer_ids)))
        
        # Adjust weights to match unique sorted layers (simplified: replicate weights logic or trust user input order?)
        # For simplicity, we assume user input is sorted and unique. If not, we might have mismatch.
        # Let's rigorously map weights to ids.
        layer_weight_map = {}
        for r, w in zip(multiscale_layers_ratio, multiscale_loss_weights):
            idx = int(r * enc_depth) - 1
            idx = max(0, min(idx, enc_depth - 1))
            layer_weight_map[idx] = w
        
        # Re-extract weights in sorted order of ids
        final_layer_weights = [layer_weight_map[i] for i in target_layer_ids]
    else:
        # Default: Last layer only
        target_layer_ids = [enc_depth - 1]
        final_layer_weights = [1.0]

    # Configure Target Encoder to output these layers
    # Note: target_encoder is MultiSeqWrapper -> backbone
    if hasattr(target_encoder, "backbone"):
        target_encoder.backbone.out_layers = target_layer_ids
    else:
        target_encoder.module.backbone.out_layers = target_layer_ids

    # Init Multi-Layer Projector
    # We assume predictor output dim == embed_dim here
    multi_layer_proj = MultiLayerProjector(pred_dim, pred_dim, target_layer_ids).to(device)
    
    # --- 新增：将 Motion Head 和 Projector 的参数加入优化器 ---
    # 我们使用与主模型相同的超参数
    optimizer.add_param_group({
        'params': motion_head.parameters(),
        'weight_decay': wd,
        'lr': lr 
    })
    optimizer.add_param_group({
        'params': multi_layer_proj.parameters(),
        'weight_decay': wd,
        'lr': lr
    })
    # -----------------------------------------

    # 仅在分布式已初始化时启用 DDP；单进程/单卡时跳过
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        encoder = DistributedDataParallel(encoder, static_graph=True)
        predictor = DistributedDataParallel(predictor, static_graph=False, find_unused_parameters=True)
        target_encoder = DistributedDataParallel(target_encoder)
        motion_head = DistributedDataParallel(motion_head)
        # Check if multi_layer_proj has parameters before wrapping with DDP
        if any(p.requires_grad for p in multi_layer_proj.parameters()):
            multi_layer_proj = DistributedDataParallel(multi_layer_proj)
    else:
        logger.info("DDP is not initialized; running without DistributedDataParallel")
    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- momentum schedule
    momentum_scheduler = (
        ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs * ipe_scale)
        for i in range(int(ipe * num_epochs * ipe_scale) + 1)
    )

    start_epoch = 0
    # -- load training checkpoint and resume training
    # import pdb; pdb.set_trace()
    if load_model or os.path.exists(latest_path):
        (
            encoder,
            predictor,
            target_encoder,
            optimizer,
            scaler,
            start_epoch,
            motion_head,
        ) = load_checkpoint_with_module_fix(
            r_path=latest_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler,
            motion_head=motion_head,
        )
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            mask_collator.step()

    def save_checkpoint(epoch, path):
        if rank != 0:
            return
        save_dict = {
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "motion_head": motion_head.state_dict(),
            "epoch": epoch,
            "loss": loss_meter.avg,
            "batch_size": batch_size,
            "world_size": world_size,
            "lr": lr,
        }
        try:
            torch.save(save_dict, path)
        except Exception as e:
            logger.info(f"Encountered exception when saving checkpoint: {e}")

    logger.info("Initializing loader...")
    unsupervised_sampler.set_epoch(start_epoch)
    loader = iter(unsupervised_loader)

    if skip_batches > 0:
        logger.info(f"Skip {skip_batches} batches")
        # -- update distributed-data-loader epoch

        for itr in range(skip_batches):
            if itr % 10 == 0:
                logger.info(f"Skip {itr}/{skip_batches} batches")
            try:
                _ = next(loader)
            except Exception:
                loader = iter(unsupervised_loader)
                _ = next(loader)

    if sync_gc:
        gc.disable()
        gc.collect()

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))

        loss_meter = AverageMeter()
        loss_jepa_meter = AverageMeter()
        loss_motion_meter = AverageMeter()
        loss_jepa_layers_meters = {}  # Dynamic dict for layer-wise losses
        mask_meters = {fpc: AverageMeter() for fpc in dataset_fpcs}
        iter_time_meter = AverageMeter()
        gpu_time_meter = AverageMeter()
        data_elapsed_time_meter = AverageMeter()

        for itr in range(ipe):
            itr_start_time = time.time()

            iter_retries = 0
            iter_successful = False
            while not iter_successful:
                try:
                    sample = next(loader)
                    iter_successful = True
                except StopIteration:
                    logger.info("Exhausted data loaders. Refreshing...")
                    unsupervised_sampler.set_epoch(epoch)
                    loader = iter(unsupervised_loader)
                except Exception as e:
                    NUM_RETRIES = 5
                    if iter_retries < NUM_RETRIES:
                        logger.warning(f"Encountered exception when loading data (num retries {iter_retries}):\n{e}")
                        iter_retries += 1
                        time.sleep(5)
                    else:
                        logger.warning(f"Exceeded max retries ({NUM_RETRIES}) when loading data. Skipping batch.")
                        raise e

            for _fpc_sample in sample:
                bs, fpc = _fpc_sample[0][-1][0].size()
                mask_meters[fpc].update(bs / batch_size)

            def load_clips():
                all_clips, all_masks_enc, all_masks_pred = [], [], []
                for fpc_sample in sample:
                    udata, masks_enc, masks_pred = fpc_sample
                    all_clips += [udata[0][0].to(device, non_blocking=True)]
                    all_masks_enc += [[m.to(device, non_blocking=True) for m in masks_enc]]
                    all_masks_pred += [[m.to(device, non_blocking=True) for m in masks_pred]]
                return all_clips, all_masks_enc, all_masks_pred

            clips, masks_enc, masks_pred = load_clips()
            data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

            if sync_gc and (itr + 1) % GARBAGE_COLLECT_ITR_FREQ == 0:
                logger.info("Running garbage collection...")
                gc.collect()

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                
                # --- 新增：计算 Ground Truth Motion ---
                with torch.no_grad():
                    # motion_target_map: [B, N_tokens_total]
                    # Compute once per batch
                    motion_target_map = get_motion_target(clips, patch_size, tubelet_size, sigma=motion_heatmap_sigma)
                # ------------------------------------
                
                def forward_target(c):
                    with torch.no_grad():
                        h = target_encoder(c)
                        # h is List[List[Tensor]] (Clips -> Layers)
                        # Normalize each layer
                        h_norm = []
                        for clip_layers in h:
                            # clip_layers: [L1, L2...]
                            normed = [F.layer_norm(lyr, (lyr.size(-1),)) for lyr in clip_layers]
                            h_norm.append(normed)
                        return h_norm

                def forward_context(c):
                    z = encoder(c, masks_enc)
                    z = predictor(z, masks_enc, masks_pred)
                    
                    # z is List[List[Tensor]] (Clips -> Predictions)
                    
                    # 1. Project to Multi-Layer Targets
                    z_multi_layers = []
                    for z_clip_preds in z:
                        z_clip_out = []
                        for z_p in z_clip_preds:
                             # z_p: [N_pred, D]
                             z_clip_out.append(multi_layer_proj(z_p))
                        z_multi_layers.append(z_clip_out)

                    # 2. Motion Head
                    z_motion_preds = []
                    for z_clip_preds in z:
                        z_motion_group = []
                        for z_p in z_clip_preds:
                            z_motion_group.append(motion_head(z_p))
                        z_motion_preds.append(z_motion_group)
                    
                    return z_multi_layers, z_motion_preds

                def loss_fn(z_layers, z_motion_preds, h, motion_target_map):
                    # h: List[List[Tensor]] (Clips -> Layers)
                    # z_layers: List[List[List[Tensor]]] (Clips -> Predictions -> Layers)
                    
                    # 1. Prepare Weights
                    motion_weights_masked = []
                    # Reuse motion map or zeros if not weighted jepa
                    if use_motion_weighted_jepa:
                         for mt, mi in zip(motion_target_map, masks_pred):
                            masked_w = apply_masks(mt.unsqueeze(-1), mi, concat=False)
                            masked_w = [m.squeeze(-1) for m in masked_w] 
                            motion_weights_masked.append(masked_w)
                    else:
                         # Fill with None to skip weighting in _compute_jepa_pair_loss
                         # Structure must match loops below
                         motion_weights_masked = [[None for _ in mi] for mi in masks_pred]
                    
                    # 2. JEPA Main Loss (Hierarchical & Weighted)
                    jepa_losses = []
                    # Dictionary to track loss per layer: {layer_idx: [loss_val, ...]}
                    jepa_layer_losses_tracker = {}
                    
                    for clip_idx, (z_preds, h_layers, mask_preds, w_preds) in enumerate(zip(z_layers, h, masks_pred, motion_weights_masked)):
                        # Iterating over CLIPS
                        for pred_idx, (z_L_list, mask, w) in enumerate(zip(z_preds, mask_preds, w_preds)):
                            # Iterating over PREDICTIONS (Masks)
                            for layer_idx, z_L in enumerate(z_L_list):
                                h_L_full = h_layers[layer_idx]
                                # Apply mask to GT
                                # apply_masks returns List, take [0]
                                h_L_masked = apply_masks(h_L_full, [mask], concat=False)[0]
                                
                                # Compute Loss using JepaLoss class
                                l_val = jepa_loss_fn(z_L, h_L_masked, motion_map=w)
                                
                                # Track raw loss per layer (before weighting) for logging
                                target_layer_id = target_layer_ids[layer_idx]
                                if target_layer_id not in jepa_layer_losses_tracker:
                                    jepa_layer_losses_tracker[target_layer_id] = []
                                jepa_layer_losses_tracker[target_layer_id].append(l_val)

                                # Apply Layer Weight
                                # final_layer_weights is available in closure
                                l_weight = final_layer_weights[layer_idx]
                                
                                jepa_losses.append(l_val * l_weight)
                    
                    if len(jepa_losses) > 0:
                        loss_jepa = torch.mean(torch.stack(jepa_losses))
                    else:
                        loss_jepa = torch.tensor(0.0, device=device, dtype=z_layers[0][0][0].dtype)
                    
                    # Aggregate layer-wise losses (mean)
                    jepa_layer_losses = {}
                    for lid, l_vals in jepa_layer_losses_tracker.items():
                        if len(l_vals) > 0:
                            jepa_layer_losses[lid] = torch.mean(torch.stack(l_vals))
                        else:
                            jepa_layer_losses[lid] = torch.tensor(0.0, device=device, dtype=loss_jepa.dtype)

                    # 3. Motion Loss (Auxiliary)
                    # Prepare motion targets (same as before)
                    # Re-calculate masked motion targets if we skipped them above for weights
                    if use_motion_weighted_jepa:
                         motion_targets_masked = motion_weights_masked
                    else:
                         # Need to compute them now
                         motion_targets_masked = []
                         for mt, mi in zip(motion_target_map, masks_pred):
                            masked = apply_masks(mt.unsqueeze(-1), mi, concat=False)  
                            masked = [m.squeeze(-1) for m in masked]  
                            motion_targets_masked.append(masked)

                    motion_losses = []
                    for pred_m_clip, target_m_clip in zip(z_motion_preds, motion_targets_masked):
                         for p_m_ij, t_m_ij in zip(pred_m_clip, target_m_clip):
                             motion_losses.append(motion_loss_fn(p_m_ij, t_m_ij))
                             
                    if len(motion_losses) > 0:
                        loss_motion = torch.mean(torch.stack(motion_losses))
                    else:
                        loss_motion = torch.tensor(0.0, device=device, dtype=z_layers[0][0][0].dtype)
                    
                    return loss_jepa + motion_loss_weight * loss_motion, loss_jepa, loss_motion, jepa_layer_losses

                # Step 1. Forward
                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    h = forward_target(clips)
                    z, z_motion_preds = forward_context(clips)
                    loss, l_jepa, l_motion, l_jepa_layers = loss_fn(z, z_motion_preds, h, motion_target_map)

                # Step 2. Backward & step

                if mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                if mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

                # Step 3. momentum update of target encoder
                m = next(momentum_scheduler)
                with torch.no_grad():
                    params_k = []
                    params_q = []
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        params_k.append(param_k)
                        params_q.append(param_q)
                    torch._foreach_mul_(params_k, m)
                    torch._foreach_add_(params_k, params_q, alpha=1 - m)

                return (
                    float(loss),
                    float(l_jepa),
                    float(l_motion),
                    {k: float(v) for k, v in l_jepa_layers.items()},
                    _new_lr,
                    _new_wd,
                )

            (
                loss,
                l_jepa,
                l_motion,
                l_jepa_layers,
                _new_lr,
                _new_wd,
            ), gpu_etime_ms = gpu_timer(train_step)
            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0
            
            # Synchronize loss across all ranks only when needed for logging
            # This minimizes communication overhead while ensuring accurate logging
            should_log = (itr % log_freq == 0) or (itr == ipe - 1) or np.isnan(loss) or np.isinf(loss)
            if should_log and torch.distributed.is_available() and torch.distributed.is_initialized():
                loss_tensor = torch.tensor(loss, device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                loss_synced = loss_tensor.item() / world_size  # Average across all ranks
            else:
                loss_synced = loss  # Use local loss when not logging
            
            # Update meter with local loss (for internal tracking)
            loss_meter.update(loss)
            loss_jepa_meter.update(l_jepa)
            loss_motion_meter.update(l_motion)
            
            # Update layer-wise meters
            for lid, l_val in l_jepa_layers.items():
                if lid not in loss_jepa_layers_meters:
                    loss_jepa_layers_meters[lid] = AverageMeter()
                loss_jepa_layers_meters[lid].update(l_val)
                
            iter_time_meter.update(iter_elapsed_time_ms)
            gpu_time_meter.update(gpu_etime_ms)
            data_elapsed_time_meter.update(data_elapsed_time_ms)

            # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, iter_elapsed_time_ms, gpu_etime_ms, data_elapsed_time_ms)
                if (itr % log_freq == 0) or (itr == ipe - 1) or np.isnan(loss) or np.isinf(loss):
                    # 计算剩余的iterations
                    remaining_iters_in_epoch = ipe - itr - 1
                    remaining_epochs = num_epochs - epoch - 1
                    total_remaining_iters = remaining_iters_in_epoch + (remaining_epochs * ipe)
                    
                    logger.info(
                        "[Epoch %d/%d, Iter %5d/%d] (Remaining: %d iters) "
                        "loss: total=%.3f jepa=%.3f motion=%.3f "
                        "masks: %s "
                        "[wd: %.2e] [lr: %.2e] "
                        "[mem: %.2e] "
                        "[iter: %.1f ms] "
                        "[gpu: %.1f ms] "
                        "[data: %.1f ms]"
                        % (
                            epoch + 1,
                            num_epochs,
                            itr + 1,  # 改为从1开始计数更直观
                            ipe,
                            total_remaining_iters,
                            loss_meter.avg,
                            loss_jepa_meter.avg,
                            loss_motion_meter.avg,
                            "[" + ", ".join([f"{k}: " + "%.1f" % mask_meters[k].avg for k in mask_meters]) + "]",
                            _new_wd,
                            _new_lr,
                            torch.cuda.max_memory_allocated() / 1024.0**2,
                            iter_time_meter.avg,
                            gpu_time_meter.avg,
                            data_elapsed_time_meter.avg,
                        )
                    )
                    
                    # Log to wandb (only on rank 0)
                    if wandb_initialized and rank == 0:
                        global_step = epoch * ipe + itr
                        wandb_metrics = {
                            "train/loss": loss_meter.avg,
                            "train/loss_jepa": loss_jepa_meter.avg,
                            "train/loss_motion": loss_motion_meter.avg,
                            "train/current_loss": loss_synced,  # Use synchronized loss for accurate logging
                            "train/lr": _new_lr,
                            "train/weight_decay": _new_wd,
                            "train/iter_time_ms": iter_time_meter.avg,
                            "train/gpu_time_ms": gpu_time_meter.avg,
                            "train/data_time_ms": data_elapsed_time_meter.avg,
                            "train/memory_mb": torch.cuda.max_memory_allocated() / 1024.0**2,
                            "train/epoch": epoch + 1,
                            "train/iteration": itr + 1,
                        }
                        # Add mask metrics
                        for k, v in mask_meters.items():
                            wandb_metrics[f"train/mask_ratio_{k}f"] = v.avg
                            
                        # Add layer-wise JEPA loss metrics
                        for lid, meter in loss_jepa_layers_meters.items():
                            wandb_metrics[f"train/loss_jepa_layer_{lid}"] = meter.avg
                            
                        wandb.log(wandb_metrics, step=global_step)

            log_stats()
            assert not np.isnan(loss), "loss is nan"

        # -- Save Checkpoint
        logger.info(
            "avg. loss total=%.3f jepa=%.3f motion=%.3f"
            % (loss_meter.avg, loss_jepa_meter.avg, loss_motion_meter.avg)
        )
        
        # Log epoch-level metrics to wandb (only on rank 0)
        if wandb_initialized and rank == 0:
            wandb.log({
                "epoch/loss": loss_meter.avg,
                "epoch": epoch + 1,
            }, step=(epoch + 1) * ipe)
        
        # -- Save Last
        if (epoch + 1) % CHECKPOINT_FREQ == 0 or epoch == (num_epochs - 1):
            # 这里不再主动清理 DataLoader，只负责安全地保存 checkpoint
            try:
                save_checkpoint(epoch + 1, latest_path)
                if save_every_freq > 0 and epoch % save_every_freq == 0:
                    save_every_file = f"e{epoch}.pt"
                    save_every_path = os.path.join(folder, save_every_file)
                    save_checkpoint(epoch + 1, save_every_path)
            except Exception as e:
                logger.error(f"Error saving checkpoint at epoch {epoch + 1}: {e}")