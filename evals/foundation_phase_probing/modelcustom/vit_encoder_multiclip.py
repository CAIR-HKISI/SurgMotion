"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
------------------------------------------------------------------------------

modelcustom API requirements:

API requirements for Encoder module:
    1) Needs to be a pytorch module with 'forward()' function protocol:
        :param x: (Tensor) Video clip (shape=[batch_size x num_channels x num_frames x height x width])
        :returns: (Tensor) Representations of video clip (shape=[batch_size x num_encoder_tokens x feature_dim])
    2) Needs to have a public attribute called 'embed_dim' (int) describing its
        output feature dimension.

API requirements for Predictor module:
    1) Needs to be a pytorch module with 'forward()' function protocol:
        :param x: (Tensor) Video clip tokens (shape=[batch_size x num_encoder_tokens x feature_dim])
        :param anticipation_time: (Tensor) Seconds into the future to predict for each sample in batch
            (shape=[batch_size])
        :returns: (Tensor) Representations of future frames (shape=[batch_size x num_output_tokens x feature_dim])
    2) Needs to have a public attribute called 'embed_dim' (int) describing its
        output feature dimension.
"""

import logging

import torch
import torch.nn as nn

import src.models.vision_transformer as vit
from src.masks.utils import apply_masks
from src.models.utils.pos_embs import get_1d_sincos_pos_embed

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def init_module(
    resolution: int,
    frames_per_clip: int,
    checkpoint: str,
    # --
    model_kwargs: dict,
    wrapper_kwargs: dict,
):
    logger.info(f"Loading pretrained model from {checkpoint}")
    checkpoint = torch.load(checkpoint, map_location="cpu")

    enc_kwargs = model_kwargs["encoder"]
    enc_ckp_key = enc_kwargs.get("checkpoint_key")
    enc_model_name = enc_kwargs.get("model_name")

    model = vit.__dict__[enc_model_name](img_size=resolution, num_frames=frames_per_clip, **enc_kwargs)

    pretrained_dict = checkpoint[enc_ckp_key]
    # --
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("encoder.", ""): v for k, v in pretrained_dict.items()}
    
    for k, v in model.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v
    msg = model.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"loaded pretrained model with msg: {msg}")
    print(model)

    model = ClipAggregation(
        model,
        tubelet_size=model.tubelet_size,
        **wrapper_kwargs,
    )
    del checkpoint
    return model


class ClipAggregation(nn.Module):
    """
    Process each clip indepdnently and concatenate all tokens
    """

    def __init__(
        self,
        model,
        tubelet_size=2,
        max_frames=128,
        use_pos_embed=False,
    ):
        super().__init__()
        self.model = model
        self.tubelet_size = tubelet_size
        self.embed_dim = embed_dim = model.embed_dim
        self.num_heads = model.num_heads

        # 1D-temporal pos-embedding
        self.pos_embed = None
        if use_pos_embed:
            max_T = max_frames // tubelet_size
            self.pos_embed = nn.Parameter(torch.zeros(1, max_T, embed_dim), requires_grad=False)
            sincos = get_1d_sincos_pos_embed(embed_dim, max_T)
            self.pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def forward(self, x, clip_indices=None):

        num_clips = len(x)
        num_views_per_clip = len(x[0])
        B, C, F, H, W = x[0][0].size()

        # Concatenate all spatial and temporal views along batch dimension
        x = [torch.cat(xi, dim=0) for xi in x]
        x = torch.cat(x, dim=0)

        outputs = self.model(x)

        def multiviews_postprocess(outputs):
            _, N, D = outputs.size()
            T = F // self.tubelet_size  # num temporal indices
            S = N // T  # num spatial tokens

            # Unroll outputs into a 2D array [spatial_views x temporal_views]
            eff_B = B * num_views_per_clip
            all_outputs = [[] for _ in range(num_views_per_clip)]
            for i in range(num_clips):
                o = outputs[i * eff_B : (i + 1) * eff_B]
                for j in range(num_views_per_clip):
                    all_outputs[j].append(o[j * B : (j + 1) * B])

            for i, outputs in enumerate(all_outputs):
                # Concatenate along temporal dimension
                outputs = [o.reshape(B, T, S, D) for o in outputs]
                outputs = torch.cat(outputs, dim=1).flatten(1, 2)
                # Compute positional embedding
                if (self.pos_embed is not None) and (clip_indices is not None):
                    _indices = [c[:, :: self.tubelet_size] for c in clip_indices]
                    pos_embed = self.pos_embed.repeat(B, 1, 1)  # [B, max_T, D]
                    pos_embed = apply_masks(pos_embed, _indices, concat=False)  # list(Tensor([B, T, D]))
                    pos_embed = torch.cat(pos_embed, dim=1)  # concatenate along temporal dimension
                    pos_embed = pos_embed.unsqueeze(2).repeat(1, 1, S, 1)  # [B, T*num_clips, S, D]
                    pos_embed = pos_embed.flatten(1, 2)
                    outputs += pos_embed
                all_outputs[i] = outputs

            return all_outputs

        return multiviews_postprocess(outputs)



import logging

import torch
import torch.nn as nn

import src.models.vision_transformer as vit
from src.masks.utils import apply_masks
from src.models.utils.pos_embs import get_1d_sincos_pos_embed

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# debug
def log_param_changes(model, logger, before_params, tag_before="加载前", tag_after="加载后", threshold=1e-5):
    """
    对比并日志输出参数在加载权重前后的变化信息。
    
    - before_params: 加载前模型参数的 state_dict
    - threshold: 数值差异阈值，超过才打印
    """
    logger.info(f"==== 参数变化对比 ({tag_before} vs {tag_after}) ====")
    changed_count = 0
    total_params = 0
    
    after_params = model.state_dict()
    
    for key in after_params:
        total_params += 1
        if key not in before_params:
            logger.info(f'【新增参数】{key}')
            changed_count += 1
            continue
        diff = (after_params[key] - before_params[key]).abs().max().item()
        if diff > threshold:
            changed_count += 1
            logger.info(f'参数 {key} 变化显著，max(abs diff)={diff:.6e}')
    
    logger.info(f"参数总数: {total_params}, 变化或新增参数数: {changed_count}")

def init_videomae_module(
    resolution: int,
    frames_per_clip: int,
    checkpoint: str,
    # --
    model_kwargs: dict,
    wrapper_kwargs: dict,
):
    logger.info(f"Loading pretrained model from {checkpoint}")
    checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    # logger.info(checkpoint['model'].keys())

    enc_kwargs = model_kwargs["encoder"]
    enc_ckp_key = enc_kwargs.get("checkpoint_key")
    enc_model_name = enc_kwargs.get("model_name")

    model = vit.__dict__[enc_model_name](img_size=resolution, num_frames=frames_per_clip, **enc_kwargs)
    # 记录加载前的参数快照
    before_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

    pretrained_dict = checkpoint[enc_ckp_key]
    # --
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("encoder.", ""): v for k, v in pretrained_dict.items()}
    for k, v in model.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v
    msg = model.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"loaded pretrained model with msg: {msg}")
    
    # 对比差异并打印
    log_param_changes(model, logger, before_state_dict, tag_before="加载前", tag_after="加载后", threshold=1e-5)
    print(model)

    model = ClipAggregation(
        model,
        tubelet_size=model.tubelet_size,
        **wrapper_kwargs,
    )
    del checkpoint
    return model


# 在evals/video_classification_frozen/modelcustom.py中新增
from transformers import AutoModel, AutoVideoProcessor
import torch

def hf_model_loader(model_name_or_path):
    """加载HuggingFace模型和处理器"""
    model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
    model.eval()  # 冻结权重
    return model

def hf_videomae_feature_extractor(model, x):
    """
    适配VideoMAE的特征提取（HuggingFace格式）
    x: 输入张量 [B, C, T, H, W]（需与模型输入格式一致）
    返回: patch特征 [B, num_patches, embed_dim]
    """
    with torch.inference_mode():
        # VideoMAE的输入是[B, T, C, H, W]，需转换维度
        x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        outputs = model(x, output_hidden_states=True)
        # 取最后一层隐藏状态，排除cls token（第0位）
        patch_features = outputs.hidden_states[-1][:, :, 1:]  # [B, T, num_patches, embed_dim]
        # 若需时序聚合，可在此添加（如取平均）
        return patch_features.mean(dim=1)  # [B, num_patches, embed_dim]

def hf_timesformer_feature_extractor(model, x):
    """适配TimeSformer的特征提取"""
    with torch.inference_mode():
        x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        outputs = model(x, output_hidden_states=True)
        # TimeSformer通常无cls token，直接取全部patch特征
        patch_features = outputs.hidden_states[-1]  # [B, T, num_patches, embed_dim]
        return patch_features.mean(dim=1)  # [B, num_patches, embed_dim]

