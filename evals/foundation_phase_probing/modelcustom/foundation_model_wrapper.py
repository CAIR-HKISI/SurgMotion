"""
Universal Foundation Model Wrapper for Attentive Probing
Supports plugging any foundation model into the attentive probing pipeline.

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
from evals.foundation_phase_probing.modelcustom.adapters.base_adapter import BaseFoundationModelAdapter

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class ClipAggregation(nn.Module):
    """
    Process each clip independently and concatenate all tokens.
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
        
        # 1D temporal positional encoding (optional)
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
        T = F // self.tubelet_size  # Number of temporal tokens
        S = N // T if N % T == 0 else N  # Number of spatial tokens
        
        # Reorganize into 2D array [spatial_views x temporal_views]
        eff_B = B * num_views_per_clip
        all_outputs = [[] for _ in range(num_views_per_clip)]
        
        for i in range(num_clips):
            o = outputs[i * eff_B : (i + 1) * eff_B]
            for j in range(num_views_per_clip):
                all_outputs[j].append(o[j * B : (j + 1) * B])
        
        # Concatenate along temporal dimension
        for i, outputs in enumerate(all_outputs):
            outputs = [o.reshape(B, T, S, D) for o in outputs]
            outputs = torch.cat(outputs, dim=1).flatten(1, 2)
            
            # Add positional encoding (optional temporal positional encoding)
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
    Generic Foundation Model initialization function.
    
    Args:
        model_type: Model type ('dinov2', 'endovit', 'endofm', 'custom', etc.)
        resolution: Image resolution
        frames_per_clip: Number of frames per clip
        checkpoint: Checkpoint path (if needed)
        model_name: Detailed model name
        wrapper_kwargs: ClipAggregation parameters
        device: Device
    
    Returns:
        ClipAggregation wrapped model
    """
    logger.info(f"="*70)
    logger.info(f"Initializing Foundation Model: {model_type}")
    logger.info(f"="*70)
    
    # Select appropriate adapter based on model_type
    if model_type == 'dinov2':
        from evals.foundation_phase_probing.modelcustom.adapters.dinov2_adapter import DINOv2Adapter
        adapter = DINOv2Adapter.from_config(
            resolution=resolution,
            checkpoint=checkpoint,
            model_name=model_name
        )
    elif model_type == 'endovit':
        from evals.foundation_phase_probing.modelcustom.adapters.endovit_adapter import EndoViTAdapter
        adapter = EndoViTAdapter.from_config(
            resolution=resolution,
            checkpoint=checkpoint,
            model_name=model_name
        )
    elif model_type == 'endofm':
        from evals.foundation_phase_probing.modelcustom.adapters.endofm_adapter import EndoFMAdapter
        adapter = EndoFMAdapter.from_config(
            resolution=resolution,
            frames_per_clip=frames_per_clip,
            checkpoint=checkpoint,
            model_name=model_name
        )
    elif model_type == 'dinov3':
        from evals.foundation_phase_probing.modelcustom.adapters.dinov3_adapter import DINOv3Adapter
        adapter = DINOv3Adapter.from_config(
            resolution=resolution,
            checkpoint=checkpoint,
            model_name=model_name
        )
    elif model_type == 'gastronet':
        from evals.foundation_phase_probing.modelcustom.adapters.gastronet_adapter import GastroNetAdapter
        adapter = GastroNetAdapter.from_config(
            resolution=resolution,
            checkpoint=checkpoint,
            model_name=model_name
        )
    elif model_type == 'selfsupsurg':
        from evals.foundation_phase_probing.modelcustom.adapters.selfsupsurg_adapter import SelfSupSurgAdapter
        adapter = SelfSupSurgAdapter.from_config(
            resolution=resolution,
            checkpoint=checkpoint,
            model_name=model_name
        )
    elif model_type == 'endossl':
        from evals.foundation_phase_probing.modelcustom.adapters.endossl_adapter import EndoSSLAdapter
        adapter = EndoSSLAdapter.from_config(
            resolution=resolution,
            checkpoint=checkpoint,
            model_name=model_name
        )
    elif model_type == 'gsvit':
        from evals.foundation_phase_probing.modelcustom.adapters.gsvit_adapter import GSViTAdapter
        adapter = GSViTAdapter.from_config(
            resolution=resolution,
            checkpoint=checkpoint,
            model_name=model_name
        )
    elif model_type == 'videomae':
        from evals.foundation_phase_probing.modelcustom.adapters.videomae_adapter import VideoMAEAdapter
        adapter = VideoMAEAdapter.from_config(
            resolution=resolution,
            frames_per_clip=frames_per_clip,
            checkpoint=checkpoint,
            model_name=model_name
        )
    elif model_type == 'surgenet':
        from evals.foundation_phase_probing.modelcustom.adapters.surgenet_adapter import SurgeNetAdapter
        adapter = SurgeNetAdapter.from_config(
            resolution=resolution,
            checkpoint=checkpoint,
            model_name=model_name
        )
    elif model_type == 'surgvlp':
        from evals.foundation_phase_probing.modelcustom.adapters.surgvlp_adapter import SurgVLPAdapter
        adapter = SurgVLPAdapter.from_config(
            resolution=resolution,
            checkpoint=checkpoint,
            model_name=model_name
        )
    elif model_type == 'internvideo':
        from evals.foundation_phase_probing.modelcustom.adapters.internvideo_adapter import InternVideoAdapter
        adapter = InternVideoAdapter.from_config(
            resolution=resolution,
            frames_per_clip=frames_per_clip,
            checkpoint=checkpoint,
            model_name=model_name
        )
    elif model_type == 'internvideo_next':
        from evals.foundation_phase_probing.modelcustom.adapters.internvideo_next_adapter import InternVideoNextAdapter
        adapter = InternVideoNextAdapter.from_config(
            resolution=resolution,
            checkpoint=checkpoint,
            model_name=model_name
        )
    elif model_type == 'endomamba':
        from evals.foundation_phase_probing.modelcustom.adapters.endomamba_adapter import EndoMambaAdapter
        adapter = EndoMambaAdapter.from_config(
            resolution=resolution,
            checkpoint=checkpoint,
            model_name=model_name or 'endomamba_small',
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Freeze adapter (for probing)
    adapter.freeze()
    
    # Wrap with ClipAggregation
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
    Standard init_module interface - compatible with the existing probing pipeline.
    
    Extracts model_type from model_kwargs, then calls init_foundation_model.
    """
    # Extract model type from model_kwargs
    model_type = model_kwargs.get('model_type', 'custom')
    
    # Extract encoder config (if available)
    encoder_kwargs = model_kwargs.get('encoder', {})
    model_name = encoder_kwargs.get('model_name', {})
    
    logger.info(f"Loading foundation model: {model_type}")
    
    # Call the generic foundation model initialization function
    model = init_foundation_model(
        model_type=model_type,
        resolution=resolution,
        frames_per_clip=frames_per_clip,
        checkpoint=checkpoint,
        model_name=model_name,  # Pass encoder config
        wrapper_kwargs=wrapper_kwargs,
        device='cpu',  # Initialize on CPU first; models.py will move to GPU
    )
    
    return model