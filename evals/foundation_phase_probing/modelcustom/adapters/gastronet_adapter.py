"""
GastroNet Foundation Model Adapter
Based on DINOv1 ViT-small architecture, uses GastroNet-5M pretrained weights
"""

import sys
sys.path.append(".")

import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from typing import Optional
from pathlib import Path

from evals.foundation_phase_probing.modelcustom.adapters.base_adapter import BaseFoundationModelAdapter

class GastroNetAdapter(BaseFoundationModelAdapter):
    """
    Adapter for GastroNet (DINOv1 ViT-small) on video input.

    Input format: [B, C, F, H, W]. Feature transform: [B, C, F, H, W] → [B, F, C, H, W] → [B*F, C, H, W],
    then backbone returns [B*F, N+1, D]; CLS removed and reshaped to [B, F*N, D].
    """
    
    def __init__(self, model, embed_dim: int, model_name: str):
        super().__init__(model, embed_dim)
        self.model_name = model_name
        self.patch_size = 16  # ViT-small standard
    
    @classmethod
    def from_config(cls, resolution: int, checkpoint: Optional[str] = None, model_name: str = 'vit_small_patch16'):
        """
        Create adapter from config, using DINOv1 ViT-small architecture.

        Args:
            resolution: Input resolution (typically 224)
            checkpoint: Checkpoint path, default is GastroNet-5M
            model_name: Model architecture name
        """
        logger.info("Loading GastroNet model: %s (DINOv1 ViT-small)", model_name)
            
        try:
            # Method 2: Load DINOv1 via torch.hub
            model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False)
            embed_dim = 384
            patch_size = 16
                
            # Load GastroNet weights
            ckpt = torch.load(
                checkpoint or "ckpts/ckpts_foundation/VITS_GastroNet-5M_DINOv1.pth",
                map_location='cpu',
                weights_only=False  # GastroNet may contain non-tensor data
            )
                
            # Handle different checkpoint formats
            if isinstance(ckpt, dict):
                if 'state_dict' in ckpt:
                    state_dict = ckpt['state_dict']
                elif 'model' in ckpt:
                    state_dict = ckpt['model']
                elif 'teacher' in ckpt:
                    state_dict = ckpt['teacher']
                else:
                    state_dict = ckpt
            else:
                state_dict = ckpt
                
            # Remove possible prefixes
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k
                # Remove common prefixes
                for prefix in ['module.', 'backbone.', 'encoder.']:
                    if new_k.startswith(prefix):
                        new_k = new_k[len(prefix):]
                new_state_dict[new_k] = v
                
            # Load weights
            msg = model.load_state_dict(new_state_dict, strict=False)
            logger.info("Loaded GastroNet weights")
            if msg.missing_keys:
                logger.debug("Missing keys: %s", len(msg.missing_keys))
            if msg.unexpected_keys:
                logger.debug("Unexpected keys: %s", len(msg.unexpected_keys))
                    
        except Exception as e2:
            logger.exception("Alternative method also failed: %s", e2)
            raise e2
        
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
               - D: embed_dim (384 for ViT-small)
        """
        B, C, F, H, W = x.shape
    
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
        
        # DINOv1 is an image model, need to unfold temporal dimension
        # [B, C, F, H, W] → [B, F, C, H, W] → [B*F, C, H, W]
        x = x.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
        x = x.reshape(B * F, C, H, W)  # [B*F, C, H, W]
        
        # Extract features via DINOv1 GastroNet
        with torch.no_grad():
            # Per DINOv1 official code, use get_intermediate_layers to get all tokens from last layer
            # This returns [B*F, N+1, D], containing CLS token and all patch tokens
            # get_intermediate_layers(x, n=1) returns output of last 1 layer
            # Return format: list of [B*F, N+1, D]
            output = self.model.get_intermediate_layers(x, n=1)
            features = output[0]  # Take first (and only): [B*F, N+1, D]
            # Remove CLS token (first position)
            features = features[:, 1:, :]  # [B*F, N, D]
        
        # Verify output shape
        if features.dim() != 3:
            raise ValueError(f"Expected 3D features [B*F, N, D], got shape: {features.shape}")
        
        BF, N, D = features.shape
        
        # Verify dimensions
        assert BF == B * F, f"Batch×Frames mismatch: {BF} != {B} * {F}"
        assert D == self.embed_dim, f"Embed dim mismatch: {D} != {self.embed_dim}"
        
        expected_num_patches = (target_H // self.patch_size) * (target_W // self.patch_size)
        assert N == expected_num_patches, (
            f"Patch number mismatch: {N} != {expected_num_patches} "
            f"(expected {target_H}/{self.patch_size} × {target_W}/{self.patch_size})"
        )
        
        # Reshape to [B, F, N, D] then flatten to [B, F*N, D]
        features = features.reshape(B, F, N, D)  # [B, F, N, D]
        features = features.reshape(B, F * N, D)  # [B, F*N, D]
        
        return features

    def get_feature_info(self):
        """Return feature extraction info (for debugging)"""
        return {
            'model_name': self.model_name,
            'embed_dim': self.embed_dim,
            'patch_size': self.patch_size,
            'architecture': 'DINOv1 ViT-small',
            'pretrain': 'GastroNet-5M',
        }


# Test code (optional)
if __name__ == "__main__":
    """
    Test GastroNet adapter input/output format

    Expected:
        Input: [B=2, C=3, F=4, H=224, W=224]
        Output: [2, 4*196, 384] = [2, 784, 384]
    """
    # Test input/output format
    adapter = GastroNetAdapter.from_config(
        resolution=224,
        checkpoint="ckpts/ckpts_foundation/VITS_GastroNet-5M_DINOv1.pth",
        model_name='vit_small_patch16'
    )
    
    # Simulated input: [B=2, C=3, F=4, H=224, W=224]
    dummy_input = torch.randn(2, 3, 4, 224, 224)
    
    print(f"\nInput shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = adapter(dummy_input)
    
    print(f"Output shape: {output.shape}")  # Should be [2, 4*196, 384]
    print(f"Expected: [2, {4 * (224//16)**2}, {adapter.embed_dim}]")
    print(f"\nFeature info: {adapter.get_feature_info()}")