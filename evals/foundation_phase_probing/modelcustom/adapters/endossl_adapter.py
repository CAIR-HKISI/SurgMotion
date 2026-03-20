"""
EndoSSL Foundation Model Adapter
Based on the EndoSSL repository, supports loading EndoSSL pretrained models
Reference: https://github.com/royhirsch/endossl
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
from evals.foundation_phase_probing.modelcustom.adapters.utils import load_and_apply_checkpoint

class EndoSSLAdapter(BaseFoundationModelAdapter):
    """
    Adapter for EndoSSL (ViT-Large) image backbone on video input.

    Input format: [B, C, F, H, W]. Feature transform: [B, C, F, H, W] → [B, F, C, H, W] → [B*F, C, H, W],
    then backbone returns [B*F, N, D] (CLS removed if present), reshaped to [B, F*N, D].
    """
    
    def __init__(self, model, embed_dim: int, model_name: str):
        super().__init__(model, embed_dim)
        self.model_name = model_name
        self.patch_size = 16  # ViT standard patch size
    
    @classmethod
    def from_config(cls, resolution: int, checkpoint: Optional[str] = None, model_name: str = 'endossl_vitl'):
        """
        Create adapter from config, load EndoSSL model.

        Args:
            resolution: Input resolution (typically 224 or 256)
            checkpoint: Checkpoint path (if None, will be auto-selected based on model_name)
            model_name: Model name identifier
                - 'endossl_laparo': Load endossl_laparo_vitl weights
                - 'endossl_colono': Load endossl_colono_vitl weights
        """
        # Map model_name to corresponding checkpoint path
        checkpoint_mapping = {
            'endossl_laparo': 'ckpts/ckpts_foundation/endossl_laparo_vitl',
            'endossl_colono': 'ckpts/ckpts_foundation/endossl_colono_vitl',
            # Default value (backward compatible)
            'endossl_vitl': 'ckpts/ckpts_foundation/endossl_laparo_vitl',
        }
        
        # If checkpoint not specified, auto-select based on model_name
        if checkpoint is None:
            if model_name in checkpoint_mapping:
                checkpoint = checkpoint_mapping[model_name]
                logger.info("Auto-selected checkpoint for '%s': %s", model_name, checkpoint)
            else:
                # If model_name not in mapping, use default
                checkpoint = checkpoint_mapping['endossl_vitl']
                logger.warning("Unknown model_name '%s', using default checkpoint: %s", model_name, checkpoint)
        
        logger.info("Loading EndoSSL model: %s", model_name)
        
        try:
            # EndoSSL typically uses ViT-Large architecture
            # Prefer local ViT implementation (consistent with other parts of project)
            try:
                from src.models.vision_transformer import vit_large
                model = vit_large(
                    patch_size=16,
                    img_size=resolution,
                    num_frames=1,  # Image model
                )
                embed_dim = 1024
                patch_size = 16
                logger.info("Created ViT-Large model using local implementation")
            except ImportError:
                # Fallback: use timm
                try:
                    import timm
                    model = timm.create_model(
                        'vit_large_patch16_224',
                        pretrained=False,
                        num_classes=0,  # Remove classification head
                        img_size=resolution
                    )
                    embed_dim = 1024
                    patch_size = 16
                    logger.info("Created ViT-Large model using timm")
                except ImportError:
                    # Last fallback: use DINOv2 architecture
                    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=False)
                    embed_dim = 1024
                    patch_size = 14
                    logger.info("Created ViT-Large model using DINOv2 architecture")
            
            logger.info("embed_dim=%s patch_size=%s resolution=%s", embed_dim, patch_size, resolution)
            
            # Check if checkpoint is file or directory
            checkpoint_path_obj = Path(checkpoint)
            if checkpoint_path_obj.is_dir():
                # If directory, find checkpoint file
                checkpoint_files = list(checkpoint_path_obj.glob("*.pth")) + \
                                 list(checkpoint_path_obj.glob("*.pt")) + \
                                 list(checkpoint_path_obj.glob("*.ckpt"))
                if checkpoint_files:
                    checkpoint_path = str(checkpoint_files[0])
                    logger.info("Found checkpoint file: %s", checkpoint_path)
                else:
                    logger.warning("No checkpoint file found in directory: %s", checkpoint_path)
                    checkpoint_path = None
            elif not checkpoint_path_obj.exists():
                logger.warning("Checkpoint not found: %s", checkpoint_path)
                checkpoint_path = None
            else:
                checkpoint_path = checkpoint
            
            # Load checkpoint
            if checkpoint_path:
                # Try different prefix removal strategies
                prefixes_to_try = [None, 'model.', 'backbone.', 'encoder.', 'module.', 'teacher.', 'student.']
                
                loaded = False
                for prefix in prefixes_to_try:
                    success, info = load_and_apply_checkpoint(
                        model=model,
                        checkpoint_path=checkpoint_path,
                        default_path=None,
                        strict=False,
                        key_prefix_to_remove=prefix,
                        verbose=(prefix is None)  # Verbose output only on first attempt
                    )
                    
                    if success:
                        if prefix:
                            logger.info("Successfully loaded with prefix '%s' removed", prefix)
                        else:
                            logger.info("Successfully loaded checkpoint")
                        loaded = True
                        break
                
                if not loaded:
                    logger.warning("Failed to load checkpoint: %s", info)
            else:
                logger.info("Using randomly initialized weights (no checkpoint loaded)")
        
        except Exception as e:
            logger.exception("Error loading EndoSSL model: %s", e)
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
        
        # Resize to model expected dimensions (typically 224)
        target_H, target_W = 224, 224
        
        # If dimensions don't match, resize
        if H != target_H or W != target_W:
            import torch.nn.functional as Fn
            # [B, C, F, H, W] → [B, F, C, H, W] → [B*F, C, H, W] → resize → [B*F, C, 224, 224]
            x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
            x_resized = Fn.interpolate(
                x_reshaped, 
                size=(target_H, target_W), 
                mode='bilinear', 
                align_corners=False
            )
            # [B*F, C, 224, 224] → [B, F, C, 224, 224] → [B, C, F, 224, 224]
            x = x_resized.reshape(B, F, C, target_H, target_W).permute(0, 2, 1, 3, 4)
            H, W = target_H, target_W
        
        # EndoSSL is an image model, need to unfold temporal dimension
        # [B, C, F, H, W] → [B, F, C, H, W] → [B*F, C, H, W]
        x = x.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
        x = x.reshape(B * F, C, H, W)  # [B*F, C, H, W]
        
        # Extract features via EndoSSL
        with torch.no_grad():
            # Extract features based on model type
            if hasattr(self.model, 'forward_features'):
                # timm or custom ViT model
                features = self.model.forward_features(x)
            elif hasattr(self.model, 'get_intermediate_layers'):
                # DINOv2 style
                output = self.model.get_intermediate_layers(x, n=1)
                features = output[0]  # [B*F, N+1, D]
            elif hasattr(self.model, 'forward'):
                # Standard ViT forward
                # Try to get patch tokens
                output = self.model(x)
                if isinstance(output, torch.Tensor):
                    features = output
                else:
                    # If returns tuple or dict, take first
                    features = output[0] if isinstance(output, (tuple, list)) else output
            else:
                raise ValueError("Cannot determine how to extract features from EndoSSL model")
            
            # Handle different output formats
            if features.dim() == 3:
                # [B*F, N+1, D] or [B*F, N, D]
                # Check if CLS token is included (typically at first position)
                if features.shape[1] == (H // self.patch_size) * (W // self.patch_size) + 1:
                    # Contains CLS token, remove it
                    features = features[:, 1:, :]  # [B*F, N, D]
            elif features.dim() == 2:
                # [B*F, D] - global features only, needs reprocessing
                raise ValueError(f"EndoSSL returned global features only, expected patch tokens. Shape: {features.shape}")
            else:
                raise ValueError(f"Unexpected feature shape: {features.shape}")
        
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
            'patch_size': self.patch_size,
            'architecture': 'ViT-Large',
            'pretrain': 'EndoSSL',
        }


# Test code
if __name__ == "__main__":
    """
    Test EndoSSL adapter input/output format

    Expected:
        Input: [B=2, C=3, F=64, H=224, W=224]
        Output: [2, 64*196, 1024] = [2, 12544, 1024]
    """
    # Test input/output format
    adapter = EndoSSLAdapter.from_config(
        resolution=224,
        checkpoint="ckpts/ckpts_foundation/endossl_colono_vitl",
        model_name='endossl_vitl'
    )
    
    # Simulated input: [B=2, C=3, F=4, H=224, W=224]
    dummy_input = torch.randn(2, 3, 4, 224, 224)
    
    print(f"\nInput shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = adapter(dummy_input)
    
    print(f"Output shape: {output.shape}")  # Should be [2, 4*196, 1024]
    print(f"Expected: [2, {4 * (224//16)**2}, {adapter.embed_dim}]")
    print(f"\nFeature info: {adapter.get_feature_info()}")