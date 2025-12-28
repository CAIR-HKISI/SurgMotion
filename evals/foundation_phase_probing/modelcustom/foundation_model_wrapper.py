"""
Universal Foundation Model Wrapper for Attentive Probing
支持任意foundation model接入到attentive probing pipeline

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
from typing import Optional, Dict, Any, Callable
from pathlib import Path
from src.masks.utils import apply_masks
from src.models.utils.pos_embs import get_1d_sincos_pos_embed
from .adapters.base_adapter import BaseFoundationModelAdapter

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class ClipAggregation(nn.Module):
    """
    Process each clip indepdnently and concatenate all tokens
    """
    
    def __init__(
        self,
        adapter: BaseFoundationModelAdapter,
        tubelet_size: int = 2,
        max_frames: int = 128,
        use_pos_embed: bool = False,
    ):
        super().__init__()
        self.adapter = adapter
        self.tubelet_size = tubelet_size
        self.embed_dim = adapter.embed_dim
        
        # 1D-temporal位置编码（可选）
        self.pos_embed = None
        if use_pos_embed:
            max_T = max_frames // tubelet_size
            self.pos_embed = nn.Parameter(
                torch.zeros(1, max_T, self.embed_dim), 
                requires_grad=False
            )
            sincos = get_1d_sincos_pos_embed(self.embed_dim, max_T)
            self.pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))
    
    def forward(self, x, clip_indices=None):
        """
        Args:
            x: List of clips, each clip is List of views
               x[i][j] has shape [B, C, F, H, W]
            clip_indices: Optional temporal indices
        Returns:
            List of aggregated features, one per view
            Each element has shape [B, T*N, D]
        """
        num_clips = len(x) #Batch size
        num_views_per_clip = len(x[0]) #Number of views per clip
        B, C, F, H, W = x[0][0].size() #Batch size, number of channels, number of frames, height, width
        
        # Concatenate all clips and views along batch dimension
        x = [torch.cat(xi, dim=0) for xi in x]
        x = torch.cat(x, dim=0)
        # x is now [B*num_clips*num_views, C, F, H, W]
        
        # Extract features through adapter
        outputs = self.adapter(x)  # [B*num_clips*num_views, N, D]

        return self._multiviews_postprocess(
            outputs, B, F, num_clips, num_views_per_clip, clip_indices
        )
    
    def _multiviews_postprocess(self, outputs, B, F, num_clips, num_views_per_clip, clip_indices):
        _, N, D = outputs.size()
        T = F // self.tubelet_size  # 时序tokens数量
        S = N // T if N % T == 0 else N  # 空间tokens数量
        
        # 重组为2D数组 [spatial_views x temporal_views]
        eff_B = B * num_views_per_clip
        all_outputs = [[] for _ in range(num_views_per_clip)]
        
        for i in range(num_clips):
            o = outputs[i * eff_B : (i + 1) * eff_B]
            for j in range(num_views_per_clip):
                all_outputs[j].append(o[j * B : (j + 1) * B])
        
        # 沿时间维度拼接
        for i, outputs in enumerate(all_outputs):
            outputs = [o.reshape(B, T, S, D) for o in outputs]
            outputs = torch.cat(outputs, dim=1).flatten(1, 2)
            
            # 添加位置编码（可选）这里可以添加temporal位置编码逻辑
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


def init_foundation_model(
    model_type: str,
    resolution: int,
    frames_per_clip: int,
    checkpoint: Optional[str] = None,
    model_name: str = None,
    wrapper_kwargs: Dict[str, Any] = None,
    device: str = "cuda",
):
    """
    通用的Foundation Model初始化函数
    
    Args:
        model_type: 模型类型 ('dinov2', 'endovit', 'endofm', 'custom', etc.)
        resolution: 图像分辨率
        frames_per_clip: 每个clip的帧数
        checkpoint: checkpoint路径（如果需要）
        model_name: 模型详细名称
        wrapper_kwargs: ClipAggregation参数
        device: 设备
    
    Returns:
        ClipAggregation wrapped model
    """
    logger.info(f"="*70)
    logger.info(f"Initializing Foundation Model: {model_type}")
    logger.info(f"="*70)
    
    # 根据model_type选择合适的adapter
    if model_type == 'dinov2':
        from .adapters.dinov2_adapter import DINOv2Adapter
        adapter = DINOv2Adapter.from_config(
            resolution=resolution,
            checkpoint=checkpoint,
            model_name=model_name
        )
    elif model_type == 'endovit':
        from .adapters.endovit_adapter import EndoViTAdapter
        adapter = EndoViTAdapter.from_config(
            resolution=resolution,
            checkpoint=checkpoint,
            model_name=model_name
        )
    elif model_type == 'endofm':
        from .adapters.endofm_adapter import EndoFMAdapter
        adapter = EndoFMAdapter.from_config(
            resolution=resolution,
            frames_per_clip=frames_per_clip,
            checkpoint=checkpoint,
            model_name=model_name
        )
    elif model_type == 'dinov3':
        from .adapters.dinov3_adapter import DINOv3Adapter
        adapter = DINOv3Adapter.from_config(
            resolution=resolution,
            checkpoint=checkpoint,
            model_name=model_name
        )
    elif model_type == 'gastronet':
        from .adapters.gastronet_adapter import GastroNetAdapter
        adapter = GastroNetAdapter.from_config(
            resolution=resolution,
            checkpoint=checkpoint,
            model_name=model_name
        )
    elif model_type == 'selfsupsurg':
        from .adapters.selfsupsurg_adapter import SelfSupSurgAdapter
        adapter = SelfSupSurgAdapter.from_config(
            resolution=resolution,
            checkpoint=checkpoint,
            model_name=model_name
        )
    elif model_type == 'endossl':
        from .adapters.endossl_adapter import EndoSSLAdapter
        adapter = EndoSSLAdapter.from_config(
            resolution=resolution,
            checkpoint=checkpoint,
            model_name=model_name
        )
    elif model_type == 'gsvit':
        from .adapters.gsvit_adapter import GSViTAdapter
        adapter = GSViTAdapter.from_config(
            resolution=resolution,
            checkpoint=checkpoint,
            model_name=model_name
        )
    elif model_type == 'videomae':
        from .adapters.videomae_adapter import VideoMAEAdapter
        adapter = VideoMAEAdapter.from_config(
            resolution=resolution,
            frames_per_clip=frames_per_clip,
            checkpoint=checkpoint,
            model_name=model_name
        )
    elif model_type == 'surgenet':
        from .adapters.surgenet_adapter import SurgeNetAdapter
        adapter = SurgeNetAdapter.from_config(
            resolution=resolution,
            checkpoint=checkpoint,
            model_name=model_name
        )
    elif model_type == 'surgvlp':
        from .adapters.surgvlp_adapter import SurgVLPAdapter
        adapter = SurgVLPAdapter.from_config(
            resolution=resolution,
            checkpoint=checkpoint,
            model_name=model_name
        )
    elif model_type == 'internvideo':
        from .adapters.internvideo_adapter import InternVideoAdapter
        adapter = InternVideoAdapter.from_config(
            resolution=resolution,
            frames_per_clip=frames_per_clip,
            checkpoint=checkpoint,
            model_name=model_name
        )
    elif model_type == 'internvideo_next':
        from .adapters.internvideo_next_adapter import InternVideoNextAdapter
        adapter = InternVideoNextAdapter.from_config(
            resolution=resolution,
            checkpoint=checkpoint,
            model_name=model_name
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # 冻结adapter（用于probing）
    adapter.freeze()
    
    # 用ClipAggregation包装
    model = ClipAggregation(
        adapter=adapter,
        tubelet_size=wrapper_kwargs.get('tubelet_size', 2),
        max_frames=wrapper_kwargs.get('max_frames', 128),
        use_pos_embed=wrapper_kwargs.get('use_pos_embed', False),
    )
    
    model = model.to(device)
    logger.info(f"✓ Foundation Model initialized with embed_dim={adapter.embed_dim}")
    logger.info(f"="*70)
    
    return model

def init_module(
    resolution: int,
    frames_per_clip: int,
    checkpoint: str,
    model_kwargs: dict,
    wrapper_kwargs: dict,
):
    """
    标准init_module接口 - 兼容现有的probing流程
    
    从model_kwargs中提取model_type，然后调用init_foundation_model
    """
    # 从model_kwargs中提取模型类型
    model_type = model_kwargs.get('model_type', 'custom')
    
    # 提取encoder配置（如果存在）
    encoder_kwargs = model_kwargs.get('encoder', {})
    model_name = encoder_kwargs.get('model_name', {})
    
    logger.info(f"Loading foundation model: {model_type}")
    
    # 调用通用的foundation model初始化函数
    model = init_foundation_model(
        model_type=model_type,
        resolution=resolution,
        frames_per_clip=frames_per_clip,
        checkpoint=checkpoint,
        model_name=model_name,  # 传递encoder配置
        wrapper_kwargs=wrapper_kwargs,
        device='cpu',  # 先在CPU上初始化，models.py会移到GPU
    )
    
    return model