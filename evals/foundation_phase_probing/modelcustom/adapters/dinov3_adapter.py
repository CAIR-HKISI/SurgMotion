"""
DINOv3 Foundation Model Adapter
Fixed version: Ensures input/output format is consistent with VisionTransformer
"""

import os
# Must be set before import transformers to prevent modelscope from hijacking from_pretrained
os.environ.setdefault("MODELSCOPE_IMPORT_PATCH", "0")

import sys
sys.path.append(".")

import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from typing import Optional

from evals.foundation_phase_probing.modelcustom.adapters.base_adapter import BaseFoundationModelAdapter


class DINOv3Adapter(BaseFoundationModelAdapter):
    """
    Adapter for DINOv3 image backbone on video input.

    Input format: [B, C, F, H, W]. Feature transform: [B, C, F, H, W] → [B, F, C, H, W] → [B*F, C, H, W],
    then backbone returns [B*F, N, D], reshaped to [B, F*N, D].
    """
    
    def __init__(self, model, embed_dim: int, model_name: str, resolution: int = 256):
        super().__init__(model, embed_dim)
        self.model_name = model_name
        self.resolution = resolution
    
    @classmethod
    def from_config(cls, resolution: int, checkpoint: Optional[str] = None, model_name: str = 'dinov3_vitl14'):
        """Create adapter from config"""
        import torch
        from transformers import AutoImageProcessor, AutoModel
                
        
        # Model path mapping
        MODEL_PATHS = {
            # Large (ViT-L/16)
            'dinov3_vitl16': "ckpts/ckpts_foundation/dinov3-vitl16-pretrain-lvd1689m",
            
            # Huge+ (ViT-H+/16)
            'dinov3_vith16plus': "ckpts/ckpts_foundation/dinov3-vith16plus-pretrain-lvd1689m", 
            
            # Giant (ViT-7B/16)
            'dinov3_vit7b16': "ckpts/ckpts_foundation/dinov3-vit7b16-pretrain-lvd1689m",
        }

        try:
            # Determine pretrained model path
            if checkpoint:
                pretrained_model_name = checkpoint
            else:
                pretrained_model_name = MODEL_PATHS.get(model_name, model_name)
            
            logger.info("Loading %s from %s", model_name, pretrained_model_name)

            processor = AutoImageProcessor.from_pretrained(pretrained_model_name, trust_remote_code=True)
            
            # Do not use device_map="auto" because evaluation framework will use DataParallel for multi-GPU
            model = AutoModel.from_pretrained(
                pretrained_model_name, 
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            # Auto-get embed_dim
            if hasattr(model.config, 'hidden_size'):
                embed_dim = model.config.hidden_size
            else:
                # Fallback
                if 'large' in model_name or 'vitl' in model_name:
                    embed_dim = 1024
                elif 'huge' in model_name or 'vith' in model_name:
                    embed_dim = 1280  # ViT-H+
                elif 'giant' in model_name or 'vit7b' in model_name:
                    embed_dim = 4096  # ViT-7B usually has larger dim, check specific config
                else:
                    embed_dim = 1024
            
            logger.info("DINOv3 loaded: embed_dim=%s", embed_dim)

        except Exception as e:
            logger.exception("Error loading DINOv3 model: %s", e)
            raise e
        
        return cls(model, embed_dim, model_name, resolution)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, F, H, W] Video input (consistent with VisionTransformer format)
               - B: batch size
               - C: channels (typically 3)
               - F: frames (temporal dimension)
               - H, W: height and width
        
        Returns:
            features: [B, F*N, D] 
               - F*N: all patch tokens from all frames concatenated
               - D: embed_dim
        """
        B, C, F, H, W = x.shape

        patch_size = getattr(self.model, 'patch_size', 16)
        target_H = self.resolution
        target_W = self.resolution
        
        # Resize if dimensions do not match
        if H != target_H or W != target_W:
            import torch.nn.functional as Fn
            # [B, C, F, H, W] → [B*F, C, H, W] → resize → [B*F, C, H', W']
            x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
            x_resized = Fn.interpolate(
                x_reshaped, 
                size=(target_H, target_W), 
                mode='bilinear', 
                align_corners=False
            )
            # [B*F, C, H', W'] → [B, F, C, H', W'] → [B, C, F, H', W']
            x = x_resized.reshape(B, F, C, target_H, target_W).permute(0, 2, 1, 3, 4)
            H, W = target_H, target_W
        
        
        # DINOv3 is an image model, need to unfold temporal dimension
        # [B, C, F, H, W] → [B, F, C, H, W] → [B*F, C, H, W]
        x = x.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)  # [B, F, C, H, W]
        
        # Extract features via DINOv3
        with torch.no_grad():
            # DINOv3 forward returns different formats depending on version
            # Try multiple ways to obtain patch tokens
            
            # Method 1: Use forward_features (recommended)
            if hasattr(self.model, 'forward_features'):
                features = self.model.forward_features(x)  # May return dict or tensor
            else:
                features = self.model(pixel_values=x, output_hidden_states=True)
            
            # Handle different output formats
            if hasattr(features, 'last_hidden_state'):
                features = features.last_hidden_state  # [B*F, N+1, D]
                features = features[:, 1:, :]

        # Validate output shape
        if features.dim() != 3:
            raise ValueError(f"Expected 3D features [B*F, N, D], got shape: {features.shape}")
        
        # Reshape back to batch and temporal dimensions
        BF, N, D = features.shape
        assert BF == B * F, f"Shape mismatch: {BF} != {B} * {F}"
        
        # Reshape to [B, F, N, D] then flatten to [B, F*N, D]
        features = features.reshape(B, F, N, D)  # [B, F, N, D]
        features = features.reshape(B, F * N, D)  # [B, F*N, D]
        
        return features


    def get_feature_info(self):
        """Return feature extraction info (for debugging)"""
        return {
            'model_name': self.model_name,
            'embed_dim': self.embed_dim,
            'patch_size': getattr(self.model, 'patch_size', 16),
        }

# Test code (optional)
if __name__ == "__main__":
    # Test input/output format
    adapter = DINOv3Adapter.from_config(
        resolution=256,
        checkpoint=None,
        model_name='dinov3_vitl16'
    )
    
    # Simulate input: [B=2, C=3, F=4, H=224, W=224]
    dummy_input = torch.randn(2, 3, 4, 256, 256)
    
    print(f"Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = adapter(dummy_input)
    
    print(f"Output shape: {output.shape}")  # Should be [2, 4*N, D]
    print(f"Expected: [2, {4 * (256//16)**2}, {adapter.embed_dim}]")
    print(f"Feature info: {adapter.get_feature_info()}")