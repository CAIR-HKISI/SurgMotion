import os
import logging
import math
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import wandb
from torch.nn.parallel import DistributedDataParallel

from evals.surgical_video_classification_frozen.models import init_module
from evals.video_classification_frozen.utils import make_transforms
from src.datasets.data_manager import init_data
from src.models.attentive_pooler import AttentiveHorizonRegressor
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.distributed import AllReduce, init_distributed
from src.utils.logging import AverageMeter

import torch.distributed as dist


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


# ----------------------
# Anticipation Metrics
# ----------------------
# 本任务统一使用 benchmark 的原始 anticipation 语义:
#   - raw target == 0.0: 当前 target 已发生 / 已出现 (present_horizon)
#   - 0.0 < raw target < horizon: target 位于 anticipation horizon 内 (inside_horizon)
#   - raw target >= horizon: target 位于 anticipation horizon 外 (outside_horizon)
#
# 训练时:
#   - 回归分支拟合 raw / horizon 后裁剪到 [0, 1] 的 normalized target
#   - 分类分支拟合上述三态 horizon state
# 评测时:
#   - MAE 在 raw 尺度上统计
#   - horizon 指标会把 prediction / target 都裁剪到 [0, horizon]
DEFAULT_ANTICIPATION_HORIZON = 5.0
HORIZON_STATE_NAMES = ["present_horizon", "outside_horizon", "inside_horizon"]


def _nanmean(values):
    """
    对一组数做 nan-safe 平均。

    - 如果列表为空，返回 nan
    - 如果全是 nan，返回 nan
    - 否则返回 np.nanmean(values)
    """
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0 or np.isnan(values).all():
        return float("nan")
    return float(np.nanmean(values))


def _masked_mae(abs_error, mask):
    """
    在 mask 指定的位置上计算 MAE。
    """
    if not np.any(mask):
        return float("nan")
    return float(np.mean(abs_error[mask]))


def _pair_mean(v1, v2):
    """
    对两个值取平均，自动忽略 nan。
    常用于 wMAE = mean(out_MAE, in_MAE)。
    """
    return _nanmean([v1, v2])


def _format_horizon_tag(horizon):
    """
    把 horizon 转成适合作为指标名前缀的字符串。

    例如:
        5.0 -> h5
        7.5 -> h7p5
    """
    if float(horizon).is_integer():
        return f"h{int(horizon)}"
    return f"h{str(horizon).replace('.', 'p')}"


def _compute_state_report(predictions_df):
    """
    生成三态分类 report。

    report 的每一行都基于真实标签(label_state)分组:
    - Acc: 该类样本中预测正确的比例
    - Support: 该类真实样本数量

    另外补一个 avg 行，表示三类 Acc 的宏平均。
    """
    if not {"pred_state", "label_state"}.issubset(predictions_df.columns):
        return None

    report = {}
    class_accs = []

    for state_name in HORIZON_STATE_NAMES:
        state_df = predictions_df[predictions_df["label_state"] == state_name]
        support = int(len(state_df))
        if support > 0:
            acc = float((state_df["pred_state"] == state_df["label_state"]).mean())
            class_accs.append(acc)
        else:
            acc = float("nan")

        report[state_name] = {
            "Acc": acc,
            "Support": support,
        }

    report["avg"] = {
        "Acc": _nanmean(class_accs),
        "Support": int(sum(v["Support"] for v in report.values())),
    }
    return report


def _classifier_num_outputs(classifier):
    module = classifier.module if hasattr(classifier, "module") else classifier
    return module.num_outputs


def _slice_target_tensor(tensor, num_targets):
    if tensor.shape[-1] == num_targets:
        return tensor
    if tensor.shape[-1] < num_targets:
        raise ValueError(
            f"标签维度 {tensor.shape[-1]} 小于当前 head 需要的 target 数 {num_targets}"
        )
    return tensor[..., :num_targets]


def _split_horizon_outputs(output, num_targets):
    num_states = len(HORIZON_STATE_NAMES)

    # 兼容旧实现: forward 直接返回 (regression, state_logits)
    if isinstance(output, tuple):
        regression, state_logits = output
        regression = _slice_target_tensor(regression, num_targets)
        if state_logits.ndim != 3:
            raise ValueError(f"state_logits 期望为 3 维张量，收到 shape={tuple(state_logits.shape)}")
        if state_logits.shape[1] == num_targets and state_logits.shape[2] == num_states:
            state_logits = state_logits.permute(0, 2, 1)
        elif state_logits.shape[1] == num_states and state_logits.shape[2] == num_targets:
            pass
        else:
            raise ValueError(
                "无法解析旧版 state_logits 形状: "
                f"shape={tuple(state_logits.shape)}, num_targets={num_targets}, num_states={num_states}"
            )
        return regression, state_logits

    if output.ndim != 2:
        raise ValueError(f"joint output 期望为 2 维张量，收到 shape={tuple(output.shape)}")

    cls_dim = num_targets * num_states
    expected_dim = cls_dim + num_targets
    if output.shape[-1] != expected_dim:
        raise ValueError(
            f"joint output 最后一维应为 {expected_dim} (= {num_states}*{num_targets}+{num_targets})，"
            f"实际收到 {output.shape[-1]}"
        )

    state_logits = output[:, :cls_dim].reshape(output.shape[0], num_states, num_targets)
    regression = output[:, cls_dim:]
    return regression, state_logits


def _ensure_target_name_column(predictions_df, target_names=None):
    """
    确保 DataFrame 中一定有 target_name 列。

    优先级:
    1. 如果本来就有 target_name，直接使用
    2. 如果没有 target_name，但有 target_idx，并且外部传了 target_names，
       就按 target_idx -> target_names 做映射
    3. 如果只提供了 target_idx，没有 target_names，
       就自动生成 target_0, target_1, ... 这样的名字

    返回:
        df: 补齐后的 DataFrame
        ordered_names: 按 target_idx 排序后的 target 名称列表
    """
    df = predictions_df.copy()

    # 已经存在 target_name，直接使用
    if "target_name" in df.columns:
        ordered_names = (
            df.sort_values("target_idx")["target_name"].drop_duplicates().tolist()
            if "target_idx" in df.columns
            else df["target_name"].drop_duplicates().tolist()
        )
        return df, ordered_names

    # 没有 target_name 时，至少需要 target_idx 来构造
    if "target_idx" not in df.columns:
        raise ValueError("predictions_df must contain either 'target_name' or 'target_idx'.")

    unique_target_idx = sorted(df["target_idx"].unique().tolist())

    # 用户显式传了 target_names，则按 target_idx 顺序映射
    if target_names is not None:
        if len(target_names) != len(unique_target_idx):
            raise ValueError(
                f"len(target_names)={len(target_names)} does not match "
                f"number of unique target_idx={len(unique_target_idx)}."
            )
        idx_to_name = dict(zip(unique_target_idx, target_names))
    else:
        # 否则自动生成名字
        idx_to_name = {idx: f"target_{idx}" for idx in unique_target_idx}

    df["target_name"] = df["target_idx"].map(idx_to_name)
    ordered_names = [idx_to_name[idx] for idx in unique_target_idx]
    return df, ordered_names


def _empty_horizon_metrics():
    return {
        "out_MAE": float("nan"),
        "in_MAE": float("nan"),
        "wMAE": float("nan"),
        "eMAE": float("nan"),
        "mMAE": float("nan"),
        "dMAE": float("nan"),
        "Clipped_MAE": float("nan"),
    }


def compute_horizon_metrics(predictions, targets, horizon):
    """
    计算给定 horizon 下的一组 anticipation 指标。

    这里采用 clipped evaluation:
    - prediction 和 target 都先裁剪到 [0, horizon]
    - 所有 horizon 相关指标都基于裁剪后的值计算
    - 因而 raw target >= horizon 的样本都会被视作 outside_horizon

    指标定义:
    - out_MAE: clipped target == horizon
      也就是原始 target >= horizon 的样本在 clip 后都会落到这里
    - in_MAE: 0 < target < horizon
    - wMAE: mean(out_MAE, in_MAE)
    - eMAE: 0 < target <= 0.1h
    - mMAE: 0.1h < target <= 0.9h
    - dMAE: 0.9h < target < h
    - Clipped_MAE: 所有样本在 clip 后的整体 MAE

    参数:
        predictions: 预测值
        targets: 真实值
        horizon: 当前评测 horizon

    返回:
        一个 dict，包含当前 horizon 下的所有指标
    """
    predictions = np.asarray(predictions, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)

    # 先把预测和标签都截断到 [0, horizon]
    clipped_predictions = np.clip(predictions, 0.0, horizon)
    clipped_targets = np.clip(targets, 0.0, horizon)

    # 绝对误差
    abs_error = np.abs(clipped_predictions - clipped_targets)

    # out 区域: clip 后等于 horizon
    out_mask = np.isclose(clipped_targets, horizon)

    # in 区域: 严格在 (0, horizon) 内
    in_mask = (clipped_targets > 0.0) & (clipped_targets < horizon)

    # 三段划分使用闭上界，避免恰好等于 0.1h / 0.9h 的样本被漏掉
    e_mask = (clipped_targets > 0.0) & (clipped_targets <= 0.1 * horizon)
    m_mask = (clipped_targets > 0.1 * horizon) & (clipped_targets <= 0.9 * horizon)
    d_mask = (clipped_targets > 0.9 * horizon) & (clipped_targets < horizon)

    out_mae = _masked_mae(abs_error, out_mask)
    in_mae = _masked_mae(abs_error, in_mask)

    return {
        "out_MAE": out_mae,
        "in_MAE": in_mae,
        "wMAE": _pair_mean(out_mae, in_mae),
        "eMAE": _masked_mae(abs_error, e_mask),
        "mMAE": _masked_mae(abs_error, m_mask),
        "dMAE": _masked_mae(abs_error, d_mask),
        "Clipped_MAE": float(abs_error.mean()) if abs_error.size > 0 else float("nan"),
    }


def compute_anticipation_metrics(predictions, targets, anticipation_horizon):
    """
    对同一组 prediction / target 计算单个 horizon 下的指标。

    参数:
        predictions: 全部预测值
        targets: 全部真实值
        anticipation_horizon: 当前任务统一使用的 anticipation horizon

    返回:
        一个扁平化的 dict，例如:
        {
            "Raw_MAE": ...,
            "h5_out_MAE": ...,
            "h5_in_MAE": ...,
            ...
            "h5_wMAE": ...
        }
    """
    predictions = np.asarray(predictions, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    anticipation_horizon = float(anticipation_horizon)

    # Raw_MAE 不做 clip，直接在原始值上计算
    metrics = {
        "Raw_MAE": float(np.mean(np.abs(predictions - targets))) if predictions.size > 0 else float("nan"),
    }

    prefix = _format_horizon_tag(anticipation_horizon)
    horizon_metrics = compute_horizon_metrics(predictions, targets, anticipation_horizon)
    for metric_name, metric_value in horizon_metrics.items():
        metrics[f"{prefix}_{metric_name}"] = metric_value

    return metrics


def evaluate_per_video_anticipation(predictions_df, target_names=None, anticipation_horizon=DEFAULT_ANTICIPATION_HORIZON):
    """
    按视频评测 anticipation 指标，并汇总整体统计。

    期望输入列:
        [data_idx, vid, target_idx, prediction, label]

    可选列:
        [target_name]

    参数:
        predictions_df: 包含预测结果的 DataFrame
        target_names: 可选，target 名称列表，用于 target_idx -> target_name 对齐
        anticipation_horizon: 当前任务统一使用的 anticipation horizon

    返回:
        per_video: 每个视频一个 dict，记录该视频的所有指标
        stats: 聚合统计结果，包括
            - 每个指标在 video 维度上的平均值
            - overall 全样本指标
            - 每个 target 的 overall 指标
            - primary horizon 信息
    """
    primary_horizon = float(anticipation_horizon)
    primary_tag = _format_horizon_tag(primary_horizon)

    # 确保 DataFrame 一定有 target_name，并得到统一顺序的 target_names
    predictions_df, target_names = _ensure_target_name_column(
        predictions_df,
        target_names=target_names,
    )

    # 按视频 / 帧 / target 排序，保证结果稳定
    sort_cols = [col for col in ["vid", "data_idx", "target_idx"] if col in predictions_df.columns]
    predictions_df = predictions_df.sort_values(sort_cols).reset_index(drop=True)

    per_video = []

    # 逐视频计算
    for vid_name, vid_data in predictions_df.groupby("vid", sort=False):
        gt = vid_data["label"].values
        pred = vid_data["prediction"].values

        metrics = compute_anticipation_metrics(pred, gt, anticipation_horizon=primary_horizon)

        # 再算该视频每个 target 在 primary horizon 下的指标
        per_target_metrics = {}
        for target_name in target_names:
            target_df = vid_data[vid_data["target_name"] == target_name]

            # 某个视频里如果某个 target 没出现，记成 nan，保持跨视频可比性
            if len(target_df) == 0:
                target_metrics = _empty_horizon_metrics()
            else:
                target_metrics = compute_horizon_metrics(
                    target_df["prediction"].values,
                    target_df["label"].values,
                    primary_horizon,
                )

            per_target_metrics[target_name] = target_metrics
            metrics[f"{target_name}_{primary_tag}_wMAE"] = target_metrics["wMAE"]

        # 当前视频所有 target 的平均 wMAE
        metrics[f"{primary_tag}_TargetAvg_wMAE"] = _nanmean(
            [v["wMAE"] for v in per_target_metrics.values()]
        )

        if {"pred_state", "label_state"}.issubset(vid_data.columns):
            metrics["StateAcc"] = float((vid_data["pred_state"] == vid_data["label_state"]).mean())

        metrics["vid"] = vid_name
        per_video.append(metrics)

    # overall: 在全部样本上直接计算一次
    all_gt = predictions_df["label"].values
    all_pred = predictions_df["prediction"].values
    overall_metrics = compute_anticipation_metrics(all_pred, all_gt, anticipation_horizon=primary_horizon)

    # overall per-target: 对每个 target 在全数据上计算主 horizon 指标
    overall_per_target = {}
    for target_name in target_names:
        target_df = predictions_df[predictions_df["target_name"] == target_name]

        if len(target_df) == 0:
            overall_per_target[target_name] = _empty_horizon_metrics()
        else:
            overall_per_target[target_name] = compute_horizon_metrics(
                target_df["prediction"].values,
                target_df["label"].values,
                primary_horizon,
            )

    overall_metrics[f"{primary_tag}_TargetAvg_wMAE"] = _nanmean(
        [v["wMAE"] for v in overall_per_target.values()]
    )
    if {"pred_state", "label_state"}.issubset(predictions_df.columns):
        overall_metrics["StateAcc"] = float(
            (predictions_df["pred_state"] == predictions_df["label_state"]).mean()
        )
        stats_state_report = _compute_state_report(predictions_df)
    else:
        stats_state_report = None

    # 需要汇总的指标键
    summary_metric_keys = ["Raw_MAE"]
    summary_metric_keys.extend([
        f"{primary_tag}_out_MAE",
        f"{primary_tag}_in_MAE",
        f"{primary_tag}_wMAE",
        f"{primary_tag}_eMAE",
        f"{primary_tag}_mMAE",
        f"{primary_tag}_dMAE",
        f"{primary_tag}_Clipped_MAE",
    ])
    summary_metric_keys.append(f"{primary_tag}_TargetAvg_wMAE")
    if "StateAcc" in overall_metrics:
        summary_metric_keys.append("StateAcc")

    stats = {}

    # 统计“每个视频先算指标，再对视频取平均”的结果
    for metric in summary_metric_keys:
        stats[f"{metric}_Mean"] = _nanmean([v.get(metric, float("nan")) for v in per_video])

    # 统计“所有样本直接合并后再算一次”的 overall 结果
    for metric in summary_metric_keys:
        stats[f"Overall_{metric}"] = overall_metrics[metric]

    # 附加信息
    stats["per_target"] = overall_per_target
    stats["primary_horizon"] = primary_horizon
    stats["primary_tag"] = primary_tag
    if stats_state_report is not None:
        stats["StateReport"] = stats_state_report

    return per_video, stats


# ----------------------
# 主入口
# ----------------------
def normalize_task_type(task_type):
    task_type = (task_type or "anticipation").lower()
    if task_type in {"anticipation", "regression", "prediction"}:
        return "anticipation"
    raise ValueError(
        f"`evals/surgical_phase_anticipation` 仅支持 anticipation/prediction，收到: '{task_type}'"
    )


def validate_anticipation_data_paths(paths, split_name):
    if isinstance(paths, str):
        paths = [paths]

    validated_paths = []
    for path in paths:
        if not isinstance(path, str):
            raise TypeError(
                f"`{split_name}` 中的每个数据路径都必须是字符串，收到: {type(path).__name__}"
            )
        if not path.endswith(".csv"):
            raise ValueError(
                f"`{split_name}` 仅接受明确的 CSV 文件路径，不再支持目录自动解析: {path}"
            )
        if not os.path.isfile(path):
            raise FileNotFoundError(f"`{split_name}` 指定的 CSV 不存在: {path}")
        validated_paths.append(path)
    return validated_paths


def infer_anticipation_output_dims(data_paths, target_names=None):
    if target_names is not None:
        return [len(target_names)] * len(data_paths)

    output_dims = []
    for data_path in data_paths:
        if not data_path.endswith(".csv"):
            raise ValueError(
                "anticipation 任务需要 CSV 数据源，或显式通过 target_names/num_classes 指定输出维度"
            )
        df = pd.read_csv(data_path, nrows=1)
        ant_cols = [c for c in df.columns if c.startswith("ant_reg_")]
        if not ant_cols:
            raise ValueError(f"{data_path} 中未找到 ant_reg_* 列，无法推断回归输出维度")
        output_dims.append(len(ant_cols))
    return output_dims


def main(args_eval, resume_preempt=False):

    val_only = args_eval.get("val_only", False)
    pretrain_folder = args_eval.get("folder", None)
    resume_checkpoint = args_eval.get("resume_checkpoint", False) or resume_preempt
    eval_tag = args_eval.get("tag", None)
    num_workers = args_eval.get("num_workers", 12)

    task_type = normalize_task_type(args_eval.get("task_type", "anticipation"))

    # wandb configuration
    wandb_config = args_eval.get("wandb", {})
    wandb_project = wandb_config.get("project", "nsjepa-surgical-probing")
    wandb_entity = wandb_config.get("entity", None)
    wandb_name = wandb_config.get("name", None)
    wandb_tags = wandb_config.get("tags", [])
    wandb_group = wandb_config.get("group", None)
    wandb_notes = wandb_config.get("notes", None)

    args_pretrain = args_eval.get("model_kwargs")
    checkpoint = args_pretrain.get("checkpoint")
    module_name = args_pretrain.get("module_name")
    args_model = args_pretrain.get("pretrain_kwargs")
    args_wrapper = args_pretrain.get("wrapper_kwargs")

    args_exp = args_eval.get("experiment")
    args_classifier = args_exp.get("classifier")
    num_probe_blocks = args_classifier.get("num_probe_blocks", 1)
    num_heads = args_classifier.get("num_heads", 16)

    args_data = args_exp.get("data")
    dataset_type = args_data.get("dataset_type", "VideoDataset")
    target_names = args_data.get("target_names")

    # Support both single and multiple datasets
    train_data_path = args_data.get("dataset_train")
    if isinstance(train_data_path, str):
        train_data_path = [train_data_path]

    val_data_path = args_data.get("dataset_val")
    if isinstance(val_data_path, str):
        val_data_path = [val_data_path]

    train_data_path = validate_anticipation_data_paths(train_data_path, "dataset_train")
    val_data_path = validate_anticipation_data_paths(val_data_path, "dataset_val")

    # Support datasets_weights for sampling from multiple datasets
    datasets_weights = args_data.get("datasets_weights", None)

    num_classes = args_data.get("num_classes")
    if isinstance(num_classes, int):
        num_classes_list = [num_classes] * len(train_data_path)
    else:
        num_classes_list = num_classes
    if num_classes_list is None:
        num_classes_list = infer_anticipation_output_dims(train_data_path, target_names=target_names)

    # Support head-to-dataset mapping (which dataset each head trains on)
    head_to_dataset_map = args_data.get("head_to_dataset_map", None)

    resolution = args_data.get("resolution", 224)
    num_segments = args_data.get("num_segments", 1)
    frames_per_clip = args_data.get("frames_per_clip", 16)
    frame_step = args_data.get("frame_step", 1)
    duration = args_data.get("clip_duration", None)
    num_views_per_segment = args_data.get("num_views_per_segment", 1)
    normalization = args_data.get("normalization", None)
    anticipation_horizon = float(
        args_data.get(
            "anticipation_horizon",
            args_eval.get("anticipation_horizon", DEFAULT_ANTICIPATION_HORIZON),
        )
    )
    args_opt = args_exp.get("optimization")
    batch_size = args_opt.get("batch_size")
    num_epochs = args_opt.get("num_epochs")
    use_bfloat16 = args_opt.get("use_bfloat16")
    horizon_ce_weight = float(args_opt.get("horizon_ce_weight", 0.1))
    
    opt_kwargs = args_opt.get("multihead_kwargs")

    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    world_size, rank = init_distributed()

    # Sync across ranks if using distributed training
    if dist.is_initialized():
        dist.barrier()

    # Initialize wandb (only on rank 0)
    if rank == 0:
        # Extract model name (parent directory of checkpoint) for logging
        if checkpoint:
            checkpoint_dir = os.path.dirname(checkpoint)
            model_name = os.path.basename(checkpoint_dir)
        else:
            model_name = "unknown"

        # Prepare config dict for wandb
        wandb_run_config = {
            "eval_name": args_eval.get("eval_name"),
            "tag": eval_tag,
            "dataset": args_eval.get("dataset", "unknown"),  # Dataset name for easy identification
            "task_type": task_type,
            "model": model_name,  # Model/checkpoint directory name
            "checkpoint_path": checkpoint,  # Full checkpoint path
            "num_workers": num_workers,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "use_bfloat16": use_bfloat16,
            "num_probe_blocks": num_probe_blocks,
            "num_heads": num_heads,
            "frames_per_clip": frames_per_clip,
            "resolution": resolution,
            "num_classes": num_classes_list,
            "target_names": target_names,
            "anticipation_horizon": anticipation_horizon,
            "horizon_ce_weight": horizon_ce_weight,
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
        logger.info(f"wandb initialized: {wandb.run.url}")

    # checkpoint 路径
    folder = pretrain_folder
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    os.makedirs(folder, exist_ok=True)
    latest_path = os.path.join(folder, "latest.pt")

    # 构建 encoder
    encoder = init_module(
        module_name=module_name,
        frames_per_clip=frames_per_clip,
        resolution=resolution,
        checkpoint=checkpoint,
        model_kwargs=args_model,
        wrapper_kwargs=args_wrapper,
        device=device,
    )

    classifiers = [
        AttentiveHorizonRegressor(
            embed_dim=encoder.embed_dim,
            num_heads=num_heads,
            depth=num_probe_blocks,
            num_outputs=(
                num_classes_list[head_to_dataset_map[idx]]
                if head_to_dataset_map is not None
                else num_classes_list[0]
            ),
            use_activation_checkpointing=True,
        ).to(device)
        for idx, _ in enumerate(opt_kwargs)
    ]

    # Only use DistributedDataParallel if distributed is initialized
    if dist.is_initialized():
        classifiers = [DistributedDataParallel(c, static_graph=True) for c in classifiers]
        logger.info(f"Wrapped {task_type} heads with DistributedDataParallel")
    else:
        logger.info(f"Running {task_type} in single-process mode (no DDP wrapping)")

    train_loader, train_sampler = make_dataloader(
        dataset_type=dataset_type,
        root_path=train_data_path,
        img_size=resolution,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        eval_duration=duration,
        num_segments=num_segments,
        num_views_per_segment=1,
        allow_segment_overlap=True,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=True,
        num_workers=num_workers,
        normalization=normalization,
        datasets_weights=datasets_weights,
        target_names=target_names,
        anticipation_horizon=anticipation_horizon,
    )
    val_loader, _ = make_dataloader(
        dataset_type=dataset_type,
        root_path=val_data_path,
        img_size=resolution,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_segments=num_segments,
        eval_duration=duration,
        num_views_per_segment=num_views_per_segment,
        allow_segment_overlap=True,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=False,
        num_workers=num_workers,
        normalization=normalization,
        target_names=target_names,
        anticipation_horizon=anticipation_horizon,
    )

    ipe = len(train_loader)

    # 多头优化器
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        classifiers=classifiers,
        opt_kwargs=opt_kwargs,
        iterations_per_epoch=ipe,
        num_epochs=num_epochs,
        use_bfloat16=use_bfloat16,
    )

    # 断点恢复
    start_epoch = 0
    if resume_checkpoint and os.path.exists(latest_path):
        encoder, classifiers, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=latest_path,
            encoder=encoder,
            classifiers=classifiers,
            opt=optimizer,
            scaler=scaler,
            val_only=val_only,
        )
        for _ in range(start_epoch * ipe):
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

    def save_checkpoint(epoch):
        save_dict = {
            "encoder": encoder.state_dict(),
            "classifiers": [c.module.state_dict() if hasattr(c, 'module') else c.state_dict() for c in classifiers],
            "opt": [o.state_dict() for o in optimizer],
            "scaler": [None if s is None else s.state_dict() for s in scaler],
            "epoch": epoch,
        }
        if rank == 0:
            torch.save(save_dict, latest_path)

    # ----------------
    # 训练循环
    # ----------------
    for epoch in range(start_epoch, num_epochs):
        train_sampler.set_epoch(epoch)

        if not val_only:
            train_metrics = run_one_epoch(
                device=device,
                training=True,
                encoder=encoder,
                classifiers=classifiers,
                scaler=scaler,
                optimizer=optimizer,
                scheduler=scheduler,
                wd_scheduler=wd_scheduler,
                data_loader=train_loader,
                use_bfloat16=use_bfloat16,
                epoch=epoch,
                head_to_dataset_map=head_to_dataset_map,
                anticipation_horizon=anticipation_horizon,
                horizon_ce_weight=horizon_ce_weight,
                rank=rank,
            )
        else:
            train_metrics = None

        val_metrics = run_one_epoch(
            device=device,
            training=False,
            encoder=encoder,
            classifiers=classifiers,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            data_loader=val_loader,
            use_bfloat16=use_bfloat16,
            epoch=epoch,
            save_predictions=True,
            folder=folder,
            head_to_dataset_map=head_to_dataset_map,
            anticipation_horizon=anticipation_horizon,
            horizon_ce_weight=horizon_ce_weight,
            rank=rank,
            train_loader_len=len(train_loader),
        )

        logger.info(f"Epoch {epoch+1}: train={train_metrics} val={val_metrics}")

        if val_only:
            if rank == 0 and wandb.run is not None:
                wandb.finish()
            if dist.is_initialized():
                dist.destroy_process_group()
            return

        save_checkpoint(epoch + 1)

    # Finish wandb run (only on rank 0)
    if rank == 0:
        wandb.finish()

    if dist.is_initialized():
        dist.destroy_process_group()


# ----------------------
# 单个 epoch 训练/验证
# ----------------------
def run_one_epoch(
    device,
    training,
    encoder,
    classifiers,
    scaler,
    optimizer,
    scheduler,
    wd_scheduler,
    data_loader,
    use_bfloat16,
    epoch=0,
    folder=None,
    save_predictions=False,
    log_interval=20,
    head_to_dataset_map=None,
    anticipation_horizon=DEFAULT_ANTICIPATION_HORIZON,
    horizon_ce_weight=0.1,
    rank=0,
    train_loader_len=None,
):
    primary_horizon = float(anticipation_horizon)
    primary_tag = _format_horizon_tag(primary_horizon)

    for c in classifiers:
        c.train(mode=training)

    base_dataset = data_loader.dataset.dataset if hasattr(data_loader.dataset, "dataset") else data_loader.dataset
    loader_target_names = getattr(base_dataset, "target_names", None)
    reg_criterion = torch.nn.MSELoss()
    state_criterion = torch.nn.CrossEntropyLoss()

    acc_meters = [AverageMeter() for _ in classifiers]
    loss_meters = [AverageMeter() for _ in classifiers]
    reg_loss_meters = [AverageMeter() for _ in classifiers]
    state_loss_meters = [AverageMeter() for _ in classifiers]
    state_acc_meters = [AverageMeter() for _ in classifiers]

    if not training:
        all_predictions = []

    for itr, data in enumerate(data_loader):
        if training:
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_bfloat16):
            clips = [[dij.to(device) for dij in di] for di in data[0]]
            clip_indices = [d.to(device) for d in data[2]]
            # 数据集返回三套 supervision:
            #   labels_norm: 归一化回归标签, 供回归损失训练
            #   horizon_states: 三态类别标签, 供 CE 分类损失训练
            #   raw_labels: 原始尺度标签, 仅用于还原 MAE / 导出预测
            labels_norm = data[1][0].to(device)
            horizon_states = data[1][1].to(device)
            raw_labels = data[1][2].to(device)
            batch_size = len(labels_norm)

            # Extract dataset_idx if available (for multi-dataset training)
            if len(data[1]) > 5:
                dataset_indices = data[1][5].to(device)
            else:
                dataset_indices = None

            with torch.no_grad():
                features = encoder(clips, clip_indices)

            outputs = [[c(f) for f in features] for c in classifiers]
            losses = []
            reg_losses = []
            state_losses = []
            has_samples = []
            labels_float = labels_norm.float()

            if head_to_dataset_map is not None and dataset_indices is not None:
                for head_idx, coutputs in enumerate(outputs):
                    assigned_dataset = head_to_dataset_map[head_idx]
                    num_targets = _classifier_num_outputs(classifiers[head_idx])
                    head_losses = []
                    head_reg_losses = []
                    head_state_losses = []
                    head_has_samples = False
                    for o in coutputs:
                        reg_pred, state_logits = _split_horizon_outputs(o, num_targets)
                        mask = (dataset_indices == assigned_dataset)
                        if mask.sum() > 0:
                            head_labels = _slice_target_tensor(labels_float[mask], num_targets)
                            head_states = _slice_target_tensor(horizon_states[mask], num_targets)
                            reg_loss = reg_criterion(reg_pred[mask], head_labels)
                            state_loss = state_criterion(
                                state_logits[mask],
                                head_states,
                            )
                            loss = reg_loss + horizon_ce_weight * state_loss
                            head_has_samples = True
                        else:
                            loss = torch.tensor(0.0, device=device, requires_grad=True)
                            reg_loss = torch.tensor(0.0, device=device)
                            state_loss = torch.tensor(0.0, device=device)
                        head_losses.append(loss)
                        head_reg_losses.append(reg_loss)
                        head_state_losses.append(state_loss)
                    losses.append(head_losses)
                    reg_losses.append(head_reg_losses)
                    state_losses.append(head_state_losses)
                    has_samples.append(head_has_samples)
            else:
                for head_idx, coutputs in enumerate(outputs):
                    num_targets = _classifier_num_outputs(classifiers[head_idx])
                    head_labels = _slice_target_tensor(labels_float, num_targets)
                    head_states = _slice_target_tensor(horizon_states, num_targets)
                    head_losses = []
                    head_reg_losses = []
                    head_state_losses = []
                    for o in coutputs:
                        reg_pred, state_logits = _split_horizon_outputs(o, num_targets)
                        reg_loss = reg_criterion(reg_pred, head_labels)
                        state_loss = state_criterion(
                            state_logits,
                            head_states,
                        )
                        head_reg_losses.append(reg_loss)
                        head_state_losses.append(state_loss)
                        head_losses.append(reg_loss + horizon_ce_weight * state_loss)
                    losses.append(head_losses)
                    reg_losses.append(head_reg_losses)
                    state_losses.append(head_state_losses)
                has_samples = [True] * len(losses)

        if training:
            if use_bfloat16:
                for s, li, o, has_sample in zip(scaler, losses, optimizer, has_samples):
                    if has_sample:
                        for lij in li:
                            s.scale(lij).backward(retain_graph=True)
                        s.step(o)
                        s.update()
                        o.zero_grad()
                    else:
                        # Skip optimizer step if no samples for this head in batch
                        o.zero_grad()
            else:
                for li, o, has_sample in zip(losses, optimizer, has_samples):
                    if has_sample:
                        for lij in li:
                            lij.backward(retain_graph=True)
                        o.step()
                        o.zero_grad()
                    else:
                        # Skip optimizer step if no samples for this head in batch
                        o.zero_grad()

        with torch.no_grad():
            for idx, coutputs in enumerate(outputs):
                num_targets = _classifier_num_outputs(classifiers[idx])
                split_outputs = [_split_horizon_outputs(o, num_targets) for o in coutputs]
                preds_norm = torch.stack([o[0] for o in split_outputs]).mean(0)
                state_logits = torch.stack([o[1] for o in split_outputs]).mean(0)
                # 回到 raw 尺度后再和 raw_labels 计算 MAE, 与 benchmark 指标语义保持一致。
                preds = torch.clamp(preds_norm, 0.0, 1.0) * anticipation_horizon
                pred_states = torch.argmax(state_logits, dim=1)

                if head_to_dataset_map is not None and dataset_indices is not None:
                    assigned_dataset = head_to_dataset_map[idx]
                    mask = (dataset_indices == assigned_dataset)
                    if mask.sum() > 0:
                        masked_labels = _slice_target_tensor(raw_labels[mask], num_targets)
                        masked_label_states = _slice_target_tensor(horizon_states[mask], num_targets)
                        masked_preds = preds[mask]
                        acc = torch.abs(masked_preds - masked_labels.float()).mean()
                        acc = float(AllReduce.apply(acc))
                        acc_meters[idx].update(acc, mask.sum().item())
                        state_acc = (pred_states[mask] == masked_label_states).float().mean()
                        state_acc = float(AllReduce.apply(state_acc))
                        state_acc_meters[idx].update(state_acc, mask.sum().item())
                else:
                    head_raw_labels = _slice_target_tensor(raw_labels, num_targets)
                    head_label_states = _slice_target_tensor(horizon_states, num_targets)
                    acc = torch.abs(preds - head_raw_labels.float()).mean()
                    acc = float(AllReduce.apply(acc))
                    acc_meters[idx].update(acc)
                    state_acc = (pred_states == head_label_states).float().mean()
                    state_acc = float(AllReduce.apply(state_acc))
                    state_acc_meters[idx].update(state_acc)

                loss_val = torch.stack([lij.detach() for lij in losses[idx]]).mean().item()
                reg_loss_val = torch.stack([rij.detach() for rij in reg_losses[idx]]).mean().item()
                state_loss_val = torch.stack([sij.detach() for sij in state_losses[idx]]).mean().item()
                loss_meters[idx].update(loss_val, batch_size)
                reg_loss_meters[idx].update(reg_loss_val, batch_size)
                state_loss_meters[idx].update(state_loss_val, batch_size)

                if not training:
                    vid_ids, data_idxs = data[1][3], data[1][4]
                    cur_target_names = loader_target_names
                    if cur_target_names is None:
                        cur_target_names = [f"target_{i}" for i in range(preds.shape[-1])]
                    else:
                        cur_target_names = cur_target_names[:preds.shape[-1]]
                    if dataset_indices is not None:
                        batch_iter = zip(
                            preds.cpu(),
                            preds_norm.cpu(),
                            raw_labels.cpu(),
                            labels_norm.cpu(),
                            pred_states.cpu(),
                            horizon_states.cpu(),
                            vid_ids.cpu(),
                            data_idxs.cpu(),
                            dataset_indices.cpu(),
                        )
                    else:
                        batch_iter = (
                            (pred, pred_norm, raw_label, norm_label, pred_state, label_state, vid, did, torch.tensor(0))
                            for pred, pred_norm, raw_label, norm_label, pred_state, label_state, vid, did in zip(
                                preds.cpu(),
                                preds_norm.cpu(),
                                raw_labels.cpu(),
                                labels_norm.cpu(),
                                pred_states.cpu(),
                                horizon_states.cpu(),
                                vid_ids.cpu(),
                                data_idxs.cpu(),
                            )
                        )
                    for pred, pred_norm, raw_label, norm_label, pred_state, label_state, vid, did, ds_idx in batch_iter:
                        for target_idx, target_name in enumerate(cur_target_names):
                            all_predictions.append([
                                idx,
                                did.item(),
                                vid.item(),
                                target_idx,
                                target_name,
                                float(pred[target_idx].item()),
                                float(raw_label[target_idx].item()),
                                ds_idx.item(),
                                float(pred_norm[target_idx].item()),
                                float(norm_label[target_idx].item()),
                                HORIZON_STATE_NAMES[int(pred_state[target_idx].item())],
                                HORIZON_STATE_NAMES[int(label_state[target_idx].item())],
                            ])

        if itr % log_interval == 0:
            if training:
                logger.info(
                    f"[Train][Epoch {epoch}][Iter {itr}/{len(data_loader)}] "
                    + " ".join(
                        [
                            f"Head{h}: MAE={am.avg:.4f}, Loss={lm.avg:.4f}, RegLoss={rm.avg:.4f}, "
                            f"StateLoss={sm.avg:.4f}, StateAcc={sam.avg:.4f}"
                            for h, (am, lm, rm, sm, sam) in enumerate(
                                zip(acc_meters, loss_meters, reg_loss_meters, state_loss_meters, state_acc_meters)
                            )
                        ]
                    )
                )

                if rank == 0:
                    global_step = epoch * len(data_loader) + itr
                    wandb_metrics = {}
                    for h, (am, lm, rm, sm, sam) in enumerate(
                        zip(acc_meters, loss_meters, reg_loss_meters, state_loss_meters, state_acc_meters)
                    ):
                        wandb_metrics[f"train/head_{h}/MAE"] = am.avg
                        wandb_metrics[f"train/head_{h}/Loss"] = lm.avg
                        wandb_metrics[f"train/head_{h}/RegLoss"] = rm.avg
                        wandb_metrics[f"train/head_{h}/StateLoss"] = sm.avg
                        wandb_metrics[f"train/head_{h}/StateAcc"] = sam.avg
                    wandb.log(wandb_metrics, step=global_step)
            else:
                logger.info(
                    f"[Val][Epoch {epoch}][Iter {itr}/{len(data_loader)}] "
                    + " ".join(
                        [
                            f"Head{h}: MAE={am.avg:.4f}, StateAcc={sam.avg:.4f}"
                            for h, (am, sam) in enumerate(zip(acc_meters, state_acc_meters))
                        ]
                    )
                )

    metrics = {
        f"head_{i}": {
            "MAE": acc_meters[i].avg,
            "Loss": loss_meters[i].avg,
            "RegLoss": reg_loss_meters[i].avg,
            "StateLoss": state_loss_meters[i].avg,
            "StateAcc": state_acc_meters[i].avg,
        }
        for i in range(len(classifiers))
    }

    if not training and len(all_predictions) > 0:
        df = pd.DataFrame(
            all_predictions,
            columns=[
                "head",
                "data_idx",
                "vid",
                "target_idx",
                "target_name",
                "prediction",
                "label",
                "dataset_idx",
                "prediction_norm",
                "label_norm",
                "pred_state",
                "label_state",
            ],
        )
        results = {}

        global_step = (epoch + 1) * (train_loader_len if train_loader_len is not None else len(data_loader))

        for head_id, g in df.groupby("head"):
            _, stats = evaluate_per_video_anticipation(
                g,
                target_names=loader_target_names,
                anticipation_horizon=primary_horizon,
            )
            results[f"head_{head_id}"] = stats

            if rank == 0:
                wandb_metrics = {}
                for metric_name in ["out_MAE", "in_MAE", "wMAE", "eMAE", "mMAE", "dMAE"]:
                    wandb_metrics[f"val/head_{head_id}/{primary_tag}/{metric_name}"] = stats[f"{primary_tag}_{metric_name}_Mean"]
                wandb_metrics[f"val/head_{head_id}/{primary_tag}/Clipped_MAE"] = stats[f"{primary_tag}_Clipped_MAE_Mean"]
                wandb_metrics[f"val/head_{head_id}/Raw_MAE"] = stats["Raw_MAE_Mean"]
                wandb_metrics[f"val/head_{head_id}/{primary_tag}/TargetAvg_wMAE"] = stats[f"{primary_tag}_TargetAvg_wMAE_Mean"]
                if "StateAcc_Mean" in stats:
                    wandb_metrics[f"val/head_{head_id}/StateAcc"] = stats["StateAcc_Mean"]
                if "StateReport" in stats:
                    wandb_metrics[f"val/head_{head_id}/StateAcc_macro_avg"] = stats["StateReport"]["avg"]["Acc"]
                    for state_name in HORIZON_STATE_NAMES:
                        wandb_metrics[f"val/head_{head_id}/StateAcc_{state_name}"] = stats["StateReport"][state_name]["Acc"]
                wandb.log(wandb_metrics, step=global_step)

        logger.info("=== Evaluation per head ===")
        for k, v in results.items():
            logger.info(
                f"{k}: "
                + " ".join(
                    [f"{primary_tag}/wMAE={v[f'{primary_tag}_wMAE_Mean']:.4f}"]
                    + ([f"StateAcc={v['StateAcc_Mean']:.4f}"] if "StateAcc_Mean" in v else [])
                )
            )
            if "per_target" in v:
                logger.info(f"  Per-target {primary_tag} wMAE for {k}:")
                for target_name, target_metrics in v["per_target"].items():
                    logger.info(f"    {target_name}: wMAE={target_metrics['wMAE']:.4f}")
            if "StateReport" in v:
                logger.info("  State classification report:")
                for state_name in HORIZON_STATE_NAMES + ["avg"]:
                    state_metrics = v["StateReport"][state_name]
                    logger.info(
                        f"    {state_name}: Acc={state_metrics['Acc']:.4f}, Support={state_metrics['Support']}"
                    )

        best_head_name = min(results.items(), key=lambda x: x[1][f"{primary_tag}_wMAE_Mean"])[0]
        best_head_stats = results[best_head_name]

        logger.info("\n" + "=" * 70)
        logger.info(f"BEST HEAD: {best_head_name} ({primary_tag}/wMAE={best_head_stats[f'{primary_tag}_wMAE_Mean']:.4f})")
        logger.info("=" * 70)
        logger.info(
            " ".join(
                [f"{primary_tag}/wMAE={best_head_stats[f'{primary_tag}_wMAE_Mean']:.4f}"]
                + (
                    [f"StateAcc={best_head_stats['StateAcc_Mean']:.4f}"]
                    if "StateAcc_Mean" in best_head_stats
                    else []
                )
            )
        )
        if "StateReport" in best_head_stats:
            logger.info("Best head state classification report:")
            for state_name in HORIZON_STATE_NAMES + ["avg"]:
                state_metrics = best_head_stats["StateReport"][state_name]
                logger.info(
                    f"  {state_name}: Acc={state_metrics['Acc']:.4f}, Support={state_metrics['Support']}"
                )
        logger.info("=" * 70)

        if rank == 0:
            wandb_best_metrics = {
                "val/best_head/Raw_MAE": best_head_stats["Raw_MAE_Mean"],
                f"val/best_head/{primary_tag}/TargetAvg_wMAE": best_head_stats[f"{primary_tag}_TargetAvg_wMAE_Mean"],
            }
            if "StateAcc_Mean" in best_head_stats:
                wandb_best_metrics["val/best_head/StateAcc"] = best_head_stats["StateAcc_Mean"]
            if "StateReport" in best_head_stats:
                wandb_best_metrics["val/best_head/StateAcc_macro_avg"] = best_head_stats["StateReport"]["avg"]["Acc"]
                for state_name in HORIZON_STATE_NAMES:
                    wandb_best_metrics[f"val/best_head/StateAcc_{state_name}"] = best_head_stats["StateReport"][state_name]["Acc"]
            for metric_name in ["out_MAE", "in_MAE", "wMAE", "eMAE", "mMAE", "dMAE"]:
                wandb_best_metrics[f"val/best_head/{primary_tag}/{metric_name}"] = best_head_stats[f"{primary_tag}_{metric_name}_Mean"]
            wandb.log(wandb_best_metrics, step=global_step)
            wandb.log({"val/best_head_id": int(best_head_name.split("_")[1])}, step=global_step)

        if head_to_dataset_map is not None:
            logger.info("\n=== Evaluation per dataset ===")
            dataset_results = {}
            for ds_idx in df["dataset_idx"].unique():
                ds_df = df[df["dataset_idx"] == ds_idx]
                assigned_heads = [h for h, d in enumerate(head_to_dataset_map) if d == ds_idx]
                logger.info(f"\nDataset {ds_idx} (Heads: {assigned_heads}):")
                for head_id in assigned_heads:
                    head_df = ds_df[ds_df["head"] == head_id]
                    if len(head_df) == 0:
                        continue

                    _, stats = evaluate_per_video_anticipation(
                        head_df,
                        target_names=loader_target_names,
                        anticipation_horizon=primary_horizon,
                    )
                    if rank == 0:
                        wandb_metrics = {}
                        wandb_metrics[f"val/dataset_{ds_idx}/head_{head_id}/{primary_tag}/wMAE"] = stats[f"{primary_tag}_wMAE_Mean"]
                        if "StateAcc_Mean" in stats:
                            wandb_metrics[f"val/dataset_{ds_idx}/head_{head_id}/StateAcc"] = stats["StateAcc_Mean"]
                        if "StateReport" in stats:
                            wandb_metrics[f"val/dataset_{ds_idx}/head_{head_id}/StateAcc_macro_avg"] = stats["StateReport"]["avg"]["Acc"]
                            for state_name in HORIZON_STATE_NAMES:
                                wandb_metrics[f"val/dataset_{ds_idx}/head_{head_id}/StateAcc_{state_name}"] = stats["StateReport"][state_name]["Acc"]
                        wandb.log(wandb_metrics, step=global_step)
                    logger.info(
                        f"  Head {head_id}: "
                        + " ".join(
                            [f"{primary_tag}/wMAE={stats[f'{primary_tag}_wMAE_Mean']:.4f}"]
                            + ([f"StateAcc={stats['StateAcc_Mean']:.4f}"] if "StateAcc_Mean" in stats else [])
                        )
                    )
                    if "StateReport" in stats:
                        logger.info("    State classification report:")
                        for state_name in HORIZON_STATE_NAMES + ["avg"]:
                            state_metrics = stats["StateReport"][state_name]
                            logger.info(
                                f"      {state_name}: Acc={state_metrics['Acc']:.4f}, Support={state_metrics['Support']}"
                            )

                    dataset_results[f"dataset_{ds_idx}_head_{head_id}"] = stats
            metrics.update(dataset_results)

        if save_predictions and folder is not None:
            pred_file = os.path.join(folder, f"predictions_epoch_{epoch}.csv")
            df.to_csv(pred_file, index=False)
            logger.info(f"Saved predictions to {pred_file}")

    return metrics


# ----------------------
# checkpoint 加载
# ----------------------
def load_checkpoint(device, r_path, encoder, classifiers, opt, scaler, val_only=False):
    checkpoint = robust_checkpoint_loader(r_path, map_location="cpu")

    encoder.load_state_dict(checkpoint["encoder"])
    for c, state in zip(classifiers, checkpoint["classifiers"]):
        if hasattr(c, 'module'):
            c.module.load_state_dict(state)
        else:
            c.load_state_dict(state)

    if val_only:
        return encoder, classifiers, opt, scaler, 0

    epoch = checkpoint["epoch"]
    for o, state in zip(opt, checkpoint["opt"]):
        o.load_state_dict(state)
    for s, state in zip(scaler, checkpoint["scaler"]):
        if s is not None and state is not None:
            s.load_state_dict(state)
    return encoder, classifiers, opt, scaler, epoch


# ----------------------
# dataloader
# ----------------------
DEFAULT_NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

def make_dataloader(
    root_path,
    batch_size,
    world_size,
    rank,
    dataset_type="VideoDataset",
    img_size=224,
    frames_per_clip=16,
    frame_step=4,
    num_segments=8,
    eval_duration=None,
    num_views_per_segment=1,
    allow_segment_overlap=True,
    training=False,
    num_workers=12,
    subset_file=None,
    normalization=None,
    datasets_weights=None,
    target_names=None,
    anticipation_horizon=DEFAULT_ANTICIPATION_HORIZON,
):
    if normalization is None:
        normalization = DEFAULT_NORMALIZATION

    transform = make_transforms(
        training=training,
        num_views_per_clip=num_views_per_segment,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(0.75, 4 / 3),
        random_resize_scale=(0.08, 1.0),
        reprob=0.25,
        auto_augment=True,
        motion_shift=False,
        crop_size=img_size,
        normalize=normalization,
    )

    data_loader, data_sampler = init_data(
        data=dataset_type,
        root_path=root_path,
        transform=transform,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        clip_len=frames_per_clip,
        frame_sample_rate=frame_step,
        duration=eval_duration,
        num_clips=num_segments,
        allow_clip_overlap=allow_segment_overlap,
        num_workers=num_workers,
        drop_last=False,
        subset_file=subset_file,
        datasets_weights=datasets_weights,
        target_names=target_names,
        anticipation_horizon=anticipation_horizon,
    )
    return data_loader, data_sampler


# ----------------------
# optimizer + scheduler
# ----------------------
def init_opt(classifiers, opt_kwargs, iterations_per_epoch, num_epochs, use_bfloat16=False):
    optimizers, schedulers, wd_schedulers, scalers = [], [], [], []
    for c, kwargs in zip(classifiers, opt_kwargs):
        param_groups = [{
            "params": c.parameters(),
            "mc_warmup_steps": int(kwargs.get("warmup") * iterations_per_epoch),
            "mc_start_lr": kwargs.get("start_lr"),
            "mc_ref_lr": kwargs.get("lr"),
            "mc_final_lr": kwargs.get("final_lr"),
            "mc_ref_wd": kwargs.get("weight_decay"),
            "mc_final_wd": kwargs.get("final_weight_decay"),
        }]
        optim = torch.optim.AdamW(param_groups)
        schedulers.append(WarmupCosineLRSchedule(optim, T_max=int(num_epochs * iterations_per_epoch)))
        wd_schedulers.append(CosineWDSchedule(optim, T_max=int(num_epochs * iterations_per_epoch)))
        optimizers.append(optim)
        scalers.append(torch.amp.GradScaler('cuda') if use_bfloat16 else None)
    return optimizers, scalers, schedulers, wd_schedulers


class WarmupCosineLRSchedule:
    def __init__(self, optimizer, T_max, last_epoch=-1):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0

    def step(self):
        self._step += 1
        for group in self.optimizer.param_groups:
            ref_lr = group.get("mc_ref_lr")
            final_lr = group.get("mc_final_lr")
            start_lr = group.get("mc_start_lr")
            warmup_steps = group.get("mc_warmup_steps")
            T_max = self.T_max - warmup_steps
            if self._step < warmup_steps:
                progress = self._step / max(1, warmup_steps)
                new_lr = start_lr + progress * (ref_lr - start_lr)
            else:
                progress = (self._step - warmup_steps) / max(1, T_max)
                new_lr = max(
                    final_lr, final_lr + (ref_lr - final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
                )
            group["lr"] = new_lr


class CosineWDSchedule:
    def __init__(self, optimizer, T_max):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0

    def step(self):
        self._step += 1
        progress = self._step / self.T_max
        for group in self.optimizer.param_groups:
            ref_wd = group.get("mc_ref_wd")
            final_wd = group.get("mc_final_wd")
            new_wd = final_wd + (ref_wd - final_wd) * 0.5 * (1.0 + math.cos(math.pi * progress))
            if final_wd <= ref_wd:
                new_wd = max(final_wd, new_wd)
            else:
                new_wd = min(final_wd, new_wd)
            group["weight_decay"] = new_wd

