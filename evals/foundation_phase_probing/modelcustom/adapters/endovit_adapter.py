"""
EndoViT Foundation Model Adapter
Based on the official EndoViT repository definition, supports loading EndoViT_SPR checkpoint
"""

import sys
sys.path.append(".")

import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from typing import Optional
from pathlib import Path
from functools import partial

from evals.foundation_phase_probing.modelcustom.adapters.base_adapter import BaseFoundationModelAdapter
from evals.foundation_phase_probing.modelcustom.adapters.utils import load_and_apply_checkpoint


class EndoViTAdapter(BaseFoundationModelAdapter):
    """
    Adapter for EndoViT image backbone on video input.

    Input format: [B, C, F, H, W]. Feature transform: [B, C, F, H, W] → [B, F, C, H, W] → [B*F, C, H, W],
    then backbone returns [B*F, N, D], reshaped to [B, F*N, D].
    """
    
    def __init__(self, model, embed_dim: int, model_name: str):
        super().__init__(model, embed_dim)
        self.model_name = model_name
    
    @classmethod
    def from_config(cls, resolution: int, checkpoint: Optional[str] = None, model_name: str = 'vit_base_patch16'):
        """
        Create adapter from config, using official EndoViT definition.

        Args:
            resolution: Input resolution (typically 224)
            checkpoint: Checkpoint path, default is EndoViT_SPR
            model_name: Model architecture name ('vit_base_patch16', 'vit_large_patch16', 'vit_huge_patch14')
        """
        import sys
        
        # Add EndoViT path to sys.path
        endovit_path = Path(__file__).parent.parent.parent.parent.parent / "foundation_models" / "EndoViT" / "pretraining" / "mae"
        if str(endovit_path) not in sys.path:
            sys.path.insert(0, str(endovit_path))
        
        # Import official model definition
        from models_vit import vit_base_patch16, vit_large_patch16, vit_huge_patch14
        
        logger.info("Loading EndoViT model: %s", model_name)
        
        try:
            model = vit_large_patch16(pool_type='no_pooling')
            embed_dim = 1024
            patch_size = 16
            
            logger.info("EndoViT model created: embed_dim=%s patch_size=%s", embed_dim, patch_size)
            
            # Use common utility to load checkpoint
            success, info = load_and_apply_checkpoint(
                model=model,
                checkpoint_path=checkpoint,
                default_path="ckpts/ckpts_foundation/endovit_SPR.pth",
                strict=False,
                key_prefix_to_remove=None,  # Set here if prefix removal is needed
                verbose=True
            )
            
            if not success:
                logger.warning("%s", info)

        except Exception as e:
            logger.exception("Error loading EndoViT model: %s", e)
            raise e
        
        return cls(model, embed_dim, model_name)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, F, H, W] video input
               - B: batch size
               - C: channels (typically 3)
               - F: frames (temporal dimension)
               - H, W: height and width

        Returns:
            features: [B, F*N, D]
               - F*N: concatenation of all patch tokens from all frames
               - D: embed_dim
        """
        B, C, F, H, W = x.shape

        # Force 224×224 as EndoViT standard input size
        target_H, target_W = 224, 224
        
        # If dimensions don't match, resize
        if H != target_H or W != target_W:
            import torch.nn.functional as fn
            # [B, C, F, H, W] → [B, F, C, H, W] → [B*F, C, H, W] → resize → [B*F, C, 224, 224]
            x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
            x_resized = fn.interpolate(x_reshaped, size=(target_H, target_W), 
                                       mode='bilinear', align_corners=False)
            # [B*F, C, 224, 224] → [B, F, C, 224, 224] → [B, C, F, 224, 224]
            x = x_resized.reshape(B, F, C, target_H, target_W).permute(0, 2, 1, 3, 4)
            H, W = target_H, target_W
        
        # EndoViT is an image model, need to unfold temporal dimension
        # [B, C, F, H, W] → [B, F, C, H, W] → [B*F, C, H, W]
        x = x.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
        x = x.reshape(B * F, C, H, W)  # [B*F, C, H, W]
        
        # Extract features via EndoViT
        with torch.no_grad():
            # Use official forward_features method
            # pool_type='no_pooling' returns [B*F, N, D] (excluding CLS token)
            features = self.model.forward_features(x)
        
        # Verify output shape
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
            'pool_type': getattr(self.model, 'pool_type', 'no_pooling'),
        }


# Test code (optional)
if __name__ == "__main__":
    """
    Input shape: torch.Size([2, 3, 4, 224, 224])
    Output shape: torch.Size([2, 784, 1024])
    Expected: [2, 784, 1024]
    """
    # Test input/output format
    adapter = EndoViTAdapter.from_config(
        resolution=224,
        checkpoint="ckpts/ckpts_foundation/endovit_SPR.pth",
        model_name='vit_base_patch16'
    )
    
    # Simulated input: [B=2, C=3, F=4, H=224, W=224]
    dummy_input = torch.randn(2, 3, 4, 224, 224)
    
    print(f"Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = adapter(dummy_input)
    
    print(f"Output shape: {output.shape}")  # Should be [2, 4*N, D]
    print(f"Expected: [2, {4 * (224//16)**2}, {adapter.embed_dim}]")
    print(f"Feature info: {adapter.get_feature_info()}")