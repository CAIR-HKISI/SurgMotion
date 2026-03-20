"""
VideoMAE/VideoMAEv2 Foundation Model Adapter
Supports loading VideoMAEv2 model via HuggingFace Transformers

Long-window support:
- Standard VideoMAE uses 16-frame input
- This adapter supports 64-frame long window via segment-based processing
- Splits 64 frames into 4 segments of 16 frames each, extracts features and concatenates
"""

import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as Fn
from typing import Optional, List

from evals.foundation_phase_probing.modelcustom.adapters.base_adapter import BaseFoundationModelAdapter


class VideoMAEAdapter(BaseFoundationModelAdapter):
    """
    VideoMAE/VideoMAEv2 model Adapter - Supports 64-frame long window
    
    Input format: [B, C, F, H, W]  (F=64 frames)
    Output format: [B, N, D] where N = num_segments * tokens_per_segment
    
    Processing strategy:
        - Split 64 frames into 4 segments of 16 frames each
        - Each segment independently extracts features through VideoMAE
        - Concatenate features from all segments
    
    Supported models:
        - OpenGVLab/VideoMAEv2-Base (768D)
        - OpenGVLab/VideoMAEv2-Large (1024D)
        - OpenGVLab/VideoMAEv2-Huge (1280D)
        - OpenGVLab/VideoMAEv2-Giant (1408D)
    """
    
    # Model configuration
    MODEL_CONFIGS = {
        'videomaev2_base': {
            'pretrained_name': 'ckpts/ckpts_foundation/OpenGVLab/VideoMAEv2-Base',
            'embed_dim': 768,
            'patch_size': 16,
            'tubelet_size': 2,
            'native_frames': 16,  # Frames natively supported by VideoMAE
        },
        'videomaev2_large': {
            'pretrained_name': 'ckpts/ckpts_foundation/OpenGVLab/VideoMAEv2-Large',
            'embed_dim': 1024,
            'patch_size': 16,
            'tubelet_size': 2,
            'native_frames': 16,
        },
        'videomaev2_huge': {
            'pretrained_name': 'ckpts/ckpts_foundation/OpenGVLab/VideoMAEv2-Huge',
            'embed_dim': 1280,
            'patch_size': 16,
            'tubelet_size': 2,
            'native_frames': 16,
        },
        'videomaev2_giant': {
            'pretrained_name': 'ckpts/ckpts_foundation/OpenGVLab/VideoMAEv2-Giant',
            'embed_dim': 1408,
            'patch_size': 14,
            'tubelet_size': 2,
            'native_frames': 16,
        },
    }
    
    def __init__(
        self, 
        model, 
        embed_dim: int, 
        model_name: str,
        processor=None,
        tubelet_size: int = 2,
        patch_size: int = 16,
        native_frames: int = 16,  # VideoMAE native frame count
        target_frames: int = 64,  # Target input frame count (long window)
    ):
        super().__init__(model, embed_dim)
        self.model_name = model_name
        self.processor = processor
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.native_frames = native_frames
        self.target_frames = target_frames
        
        # Compute number of segments
        self.num_segments = target_frames // native_frames
        if target_frames % native_frames != 0:
            logger.warning(
                "target_frames (%s) not divisible by native_frames (%s), num_segments=%s",
                target_frames, native_frames, (target_frames + native_frames - 1) // native_frames
            )
            self.num_segments = (target_frames + native_frames - 1) // native_frames

    @classmethod
    def from_config(
        cls, 
        resolution: int, 
        frames_per_clip: int = 64,  # Default 64-frame long window
        checkpoint: Optional[str] = None, 
        model_name: str = 'videomaev2_large'
    ):
        """
        Create adapter from config
        
        Args:
            resolution: Input resolution (typically 224)
            frames_per_clip: Frames per clip (default 64-frame long window)
            checkpoint: Custom checkpoint path (optional)
            model_name: Model name, supports:
                - 'videomaev2_base', 'videomaev2_large', 'videomaev2_huge', 'videomaev2_giant'
        """
        from transformers import VideoMAEImageProcessor, AutoModel, AutoConfig
        
        logger.info("Loading VideoMAE model: %s", model_name)
        
        # Get model configuration
        if model_name in cls.MODEL_CONFIGS:
            config_info = cls.MODEL_CONFIGS[model_name]
            pretrained_name = config_info['pretrained_name']
            embed_dim = config_info['embed_dim']
            patch_size = config_info['patch_size']
            tubelet_size = config_info['tubelet_size']
            native_frames = config_info['native_frames']
        else:
            # Allow direct use of HuggingFace model name
            pretrained_name = model_name
            embed_dim = 1024
            patch_size = 16
            tubelet_size = 2
            native_frames = 16
        
        try:
            # Load config
            config = AutoConfig.from_pretrained(pretrained_name, trust_remote_code=True, low_cpu_mem_usage=False)
            
            # Update embed_dim (get more accurate value from config)
            if hasattr(config, 'hidden_size'):
                embed_dim = config.hidden_size
            
            # Load processor
            processor = VideoMAEImageProcessor.from_pretrained(pretrained_name)
            
            # Load model
            if checkpoint and checkpoint.lower() != 'none':
                logger.info("Loading from custom checkpoint: %s", checkpoint)
                model = AutoModel.from_pretrained(
                    pretrained_name, 
                    config=config, 
                    trust_remote_code=True,
                    low_cpu_mem_usage=False
                )
                # Load custom weights
                state_dict = torch.load(checkpoint, map_location='cpu')
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                elif 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                
                # Remove possible prefixes
                for prefix in ['module.', 'backbone.', 'encoder.']:
                    state_dict = {k.replace(prefix, ''): v for k, v in state_dict.items()}
                
                model.load_state_dict(state_dict, strict=False)
                logger.info("Loaded custom checkpoint")
            else:
                model = AutoModel.from_pretrained(
                    pretrained_name, 
                    config=config, 
                    trust_remote_code=True,
                    low_cpu_mem_usage=False
                )
            
            logger.info(
                "VideoMAE loaded: pretrained=%s embed_dim=%s patch_size=%s tubelet_size=%s native_frames=%s target_frames=%s",
                pretrained_name, embed_dim, patch_size, tubelet_size, native_frames, frames_per_clip
            )
            
        except Exception as e:
            logger.exception("Error loading VideoMAE model: %s", e)
            raise e
        
        return cls(
            model=model, 
            embed_dim=embed_dim, 
            model_name=model_name,
            processor=processor,
            tubelet_size=tubelet_size,
            patch_size=patch_size,
            native_frames=native_frames,
            target_frames=frames_per_clip,
        )
    
    def _adjust_temporal_length(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adjust temporal length to target_frames
        
        Args:
            x: [B, C, F, H, W]
        Returns:
            x: [B, C, target_frames, H, W]
        """
        B, C, F, H, W = x.shape
        
        if F == self.target_frames:
            return x
        
        if F > self.target_frames:
            # Uniform sampling
            indices = torch.linspace(0, F - 1, self.target_frames).long()
            x = x[:, :, indices, :, :]
        else:
            # Repeat padding
            repeat_times = (self.target_frames + F - 1) // F
            x = x.repeat(1, 1, repeat_times, 1, 1)[:, :, :self.target_frames, :, :]
        
        return x
    
    def _adjust_spatial_size(self, x: torch.Tensor, target_size: int = 224) -> torch.Tensor:
        """
        Adjust spatial resolution
        
        Args:
            x: [B, C, F, H, W]
            target_size: Target resolution
        Returns:
            x: [B, C, F, target_size, target_size]
        """
        B, C, F, H, W = x.shape
        
        if H == target_size and W == target_size:
            return x
        
        # [B, C, F, H, W] → [B, F, C, H, W] → [B*F, C, H, W] → resize → [B, C, F, H', W']
        x = x.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
        x = Fn.interpolate(x, size=(target_size, target_size), mode='bilinear', align_corners=False)
        x = x.reshape(B, F, C, target_size, target_size).permute(0, 2, 1, 3, 4)
        return x
    
    def _split_into_segments(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Split long video into multiple segments
        
        Args:
            x: [B, C, F, H, W] where F = target_frames (64)
        Returns:
            segments: List of [B, C, native_frames, H, W] (4 segments)
        """
        B, C, F, H, W = x.shape
        segments = []
        
        for i in range(self.num_segments):
            start_idx = i * self.native_frames
            end_idx = min(start_idx + self.native_frames, F)
            
            segment = x[:, :, start_idx:end_idx, :, :]
            
            # If last segment has fewer than native_frames, pad
            if segment.shape[2] < self.native_frames:
                pad_size = self.native_frames - segment.shape[2]
                # Repeat last frame for padding
                padding = segment[:, :, -1:, :, :].repeat(1, 1, pad_size, 1, 1)
                segment = torch.cat([segment, padding], dim=2)
            
            segments.append(segment)
        
        return segments
    
    def _extract_segment_features(self, segment: torch.Tensor) -> torch.Tensor:
        """
        Extract features for a single segment (patch tokens, not pooled)
        
        Args:
            segment: [B, C, native_frames, H, W]
        Returns:
            features: [B, N, D] where N = (native_frames/tubelet) * (H/patch) * (W/patch)
        """
        # VideoMAEv2 (OpenGVLab) defaults to mean pooling for pooled features
        # We need patch tokens, access internal VisionTransformer directly
        
        # Get internal VisionTransformer model
        vit_model = self._get_inner_vit()
        
        # Manually extract patch tokens (bypass mean pooling)
        features = self._manual_extract_patch_tokens(vit_model, segment)
        
        # Ensure 3D tensor [B, N, D]
        if features.dim() == 2:
            features = features.unsqueeze(1)
        elif features.dim() != 3:
            raise ValueError(f"Expected 2D or 3D features, got shape: {features.shape}")
        
        return features
    
    def _get_inner_vit(self):
        """Get internal VisionTransformer model"""
        # OpenGVLab VideoMAEv2 structure: VideoMAEv2 -> model (VisionTransformer)
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'patch_embed'):
            return self.model.model
        elif hasattr(self.model, 'patch_embed'):
            return self.model
        elif hasattr(self.model, 'videomae'):
            return self.model.videomae
        elif hasattr(self.model, 'encoder'):
            return self.model.encoder
        else:
            raise ValueError("Cannot find inner VisionTransformer")
    
    def _manual_extract_patch_tokens(self, vit: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        Manually extract VideoMAEv2 patch tokens (bypass mean pooling)
        
        Model structure:
            patch_embed -> pos_drop -> blocks -> fc_norm
        
        Args:
            vit: VisionTransformer model
            x: [B, C, T, H, W]
        Returns:
            features: [B, N, D] all patch tokens
        """
        B = x.shape[0]
        
        # 1. Patch embedding: [B, C, T, H, W] -> [B, N, D]
        x = vit.patch_embed(x)
        
        # 2. Add position encoding (if using learnable pos embed)
        if hasattr(vit, 'pos_embed') and vit.pos_embed is not None:
            # Ensure pos_embed is on the correct device
            if vit.pos_embed.device != x.device:
                x = x + vit.pos_embed.to(x.device)
            else:
                x = x + vit.pos_embed
        
        # 3. Position dropout
        if hasattr(vit, 'pos_drop'):
            x = vit.pos_drop(x)
        
        # 4. Pass through all Transformer blocks
        for blk in vit.blocks:
            x = blk(x)
        
        # 5. Final norm (fc_norm instead of mean pooling)
        # Note: Do not use mean pooling, keep all patch tokens
        if hasattr(vit, 'fc_norm') and vit.fc_norm is not None:
            x = vit.fc_norm(x)
        elif hasattr(vit, 'norm') and vit.norm is not None:
            x = vit.norm(x)
        
        return x  # [B, N, D]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - supports 64-frame long window
        
        Args:
            x: [B, C, F, H, W] Video input
               - B: batch size
               - C: channels (typically 3)
               - F: frames (64-frame long window)
               - H, W: height and width
        
        Returns:
            features: [B, N_total, D] 
               - N_total = num_segments * N_per_segment
               - N_per_segment = (native_frames/tubelet) * (H/patch) * (W/patch)
               - D: embed_dim
        """
        B, C, F, H, W = x.shape
        
        # 1. Adjust temporal length to target_frames (64)
        x = self._adjust_temporal_length(x)
        
        # 2. Adjust spatial resolution to 224x224
        target_spatial = 224
        if H != target_spatial or W != target_spatial:
            x = self._adjust_spatial_size(x, target_spatial)
            H, W = target_spatial, target_spatial
        
        # 3. Segment processing
        segments = self._split_into_segments(x)  # List of [B, C, 16, H, W]
        
        # 4. Extract features for each segment
        all_features = []
        with torch.no_grad():
            for i, segment in enumerate(segments):
                features = self._extract_segment_features(segment)  # [B, N, D]
                all_features.append(features)
        
        # 5. Concatenate features from all segments
        # [B, N, D] * num_segments -> [B, N_total, D]
        features = torch.cat(all_features, dim=1)
        
        # Validate output
        if features.dim() != 3:
            raise ValueError(f"Expected 3D features [B, N, D], got shape: {features.shape}")
        
        return features
    
    def get_feature_info(self):
        """Return feature extraction info (for debugging)"""
        # Compute token count
        tokens_per_segment = (
            (self.native_frames // self.tubelet_size) * 
            (224 // self.patch_size) * 
            (224 // self.patch_size)
        )
        total_tokens = tokens_per_segment * self.num_segments
        
        return {
            'model_name': self.model_name,
            'embed_dim': self.embed_dim,
            'patch_size': self.patch_size,
            'tubelet_size': self.tubelet_size,
            'native_frames': self.native_frames,
            'target_frames': self.target_frames,
            'num_segments': self.num_segments,
            'tokens_per_segment': tokens_per_segment,
            'total_tokens': total_tokens,
        }


# Test code
if __name__ == "__main__":
    import numpy as np
    print("Testing VideoMAEAdapter with 64-frame Long Window")
    
    # Create adapter (64-frame long window)
    adapter = VideoMAEAdapter.from_config(
        resolution=224,
        frames_per_clip=64,  # 64-frame long window
        checkpoint=None,
        model_name='videomaev2_large'
    )
    
    # Freeze parameters
    adapter.freeze()
    
    # Test input: [B=2, C=3, F=64, H=224, W=224]
    dummy_input = torch.randn(2, 3, 64, 224, 224)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Feature info: {adapter.get_feature_info()}")
    
    with torch.no_grad():
        output = adapter(dummy_input)
    
    print(f"\nOutput shape: {output.shape}")
    
    # Verify
    info = adapter.get_feature_info()
    expected_tokens = info['total_tokens']
    print(f"Expected shape: [2, {expected_tokens}, {adapter.embed_dim}]")
    
    assert output.shape == (2, expected_tokens, adapter.embed_dim), \
        f"Shape mismatch! Got {output.shape}, expected (2, {expected_tokens}, {adapter.embed_dim})"
    
    print("\n✓ VideoMAEAdapter 64-frame long window test passed!")
