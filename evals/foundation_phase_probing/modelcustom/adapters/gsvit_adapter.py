"""
GSViT Foundation Model Adapter
Based on GSViT (General Surgery Vision Transformer) model
Reference: https://github.com/royhirsch/GSViT
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


class GSViTAdapter(BaseFoundationModelAdapter):
    """
    Adapter for GSViT (EfficientViT-M5) image backbone on video input.

    Input format: [B, C, F, H, W]. Feature transform: [B, C, F, H, W] → [B, F, C, H, W] → [B*F, C, H, W];
    backbone returns feature map [B*F, 384, 4, 4], flattened to [B, F*N, D] (N=16 per frame).
    """
    
    def __init__(self, model, embed_dim: int, model_name: str):
        super().__init__(model, embed_dim)
        self.model_name = model_name
        self.patch_size = 16  # Standard EfficientViT patch size
    
    @classmethod
    def from_config(cls, resolution: int, checkpoint: Optional[str] = None, model_name: str = 'gsvit'):
        """
        Create adapter from config, load GSViT model
        
        Args:
            resolution: Input resolution (typically 224)
            checkpoint: Path to checkpoint
            model_name: Model name identifier
        """
        import sys
        
        # Add GSViT path to sys.path
        gsvit_path = Path(__file__).parent.parent.parent.parent.parent / "foundation_models" / "GSViT"
        if str(gsvit_path) not in sys.path:
            sys.path.insert(0, str(gsvit_path))
        
        logger.info("Loading GSViT model: %s", model_name)
        
        try:
            # Import GSViT model definition
            from EfficientViT.classification.model.build import EfficientViT_M5
            
            # Create EfficientViT_M5 model (without classification head)
            # GSViT is based on EfficientViT_M5; removing the classification head leaves only the feature extraction part
            evit_model = EfficientViT_M5(pretrained=False, num_classes=0)
            # Remove classification head (if present)
            if hasattr(evit_model, 'head'):
                evit_model.head = nn.Identity()
            
            # Final layer embed_dim of EfficientViT_M5 is 384
            embed_dim = 384
            patch_size = 16
            
            logger.info("GSViT architecture created: embed_dim=%s patch_size=%s resolution=%s", embed_dim, patch_size, resolution)
            
            # Load checkpoint
            checkpoint_path = checkpoint or "ckpts/ckpts_foundation/GSViT.pkl"
            
            if not Path(checkpoint_path).exists():
                logger.warning("Checkpoint not found: %s", checkpoint_path)
                checkpoint_path = None
            
            if checkpoint_path:
                logger.info("Loading checkpoint from: %s", checkpoint_path)
                
                # GSViT checkpoint format: contains entire EfficientViT model (with classification head removed)
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                    
                    # Handle different checkpoint formats
                    if isinstance(checkpoint, dict):
                        # Check if specific key exists
                        if 'evit' in checkpoint:
                            state_dict = checkpoint['evit']
                        elif 'model' in checkpoint:
                            state_dict = checkpoint['model']
                        elif 'state_dict' in checkpoint:
                            state_dict = checkpoint['state_dict']
                        else:
                            state_dict = checkpoint
                    else:
                        state_dict = checkpoint
                    
                    # Try to load weights
                    # GSViT checkpoint may contain complete EfficientViT model (with head removed)
                    # Need to match model structure
                    try:
                        # Try direct load
                        msg = evit_model.load_state_dict(state_dict, strict=False)
                        logger.info("Checkpoint loaded")
                        if msg.missing_keys:
                            logger.debug("Missing keys: %s", len(msg.missing_keys))
                        if msg.unexpected_keys:
                            logger.debug("Unexpected keys: %s", len(msg.unexpected_keys))
                    except Exception as e1:
                        # If direct load fails, try removing prefix
                        logger.debug("Trying to load with prefix removal")
                        cleaned_state_dict = {}
                        for k, v in state_dict.items():
                            # Try to remove common prefixes
                            new_key = k
                            for prefix in ['evit.', 'model.', 'backbone.', 'encoder.', 'module.']:
                                if new_key.startswith(prefix):
                                    new_key = new_key[len(prefix):]
                                    break
                            cleaned_state_dict[new_key] = v
                        
                        msg = evit_model.load_state_dict(cleaned_state_dict, strict=False)
                        logger.info("Checkpoint loaded with prefix removal")
                        if msg.missing_keys:
                            logger.debug("Missing keys: %s", len(msg.missing_keys))
                        if msg.unexpected_keys:
                            logger.debug("Unexpected keys: %s", len(msg.unexpected_keys))
                
                except Exception as e:
                    logger.warning("Failed to load checkpoint: %s, using randomly initialized weights", e)
            else:
                logger.info("Using randomly initialized weights (no checkpoint loaded)")
            
            # Wrap model to handle input (color channel flip)
            model = GSViTWrapper(evit_model)
        
        except Exception as e:
            logger.exception("Error loading GSViT model: %s", e)
            raise e
        
        return cls(model, embed_dim, model_name)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, F, H, W] video input
               - B: batch size, C: channels (3), F: frames, H, W: height and width

        Returns:
            features: [B, F*N, D]
               - F*N: all spatial positions from all frames (N=16 per frame); D: embed_dim (384)
        """
        B, C, F, H, W = x.shape
        
        target_H, target_W = 224, 224
        if H != target_H or W != target_W:
            import torch.nn.functional as Fn
            # [B, C, F, H, W] → [B, F, C, H, W] → [B*F, C, H, W] → resize
            x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
            x_resized = Fn.interpolate(
                x_reshaped, 
                size=(target_H, target_W), 
                mode='bilinear', 
                align_corners=False
            )
            x = x_resized.reshape(B, F, C, target_H, target_W).permute(0, 2, 1, 3, 4)
            H, W = target_H, target_W
        
        # [B, C, F, H, W] → [B, F, C, H, W] → [B*F, C, H, W]
        x = x.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
        
        # Extract features via GSViT
        with torch.no_grad():
            # GSViT outputs feature map [B*F, 384, 4, 4]
            feature_map = self.model(x)
        
        # Convert feature map to patch tokens format
        # [B*F, 384, 4, 4] -> [B*F, 16, 384]
        BF, C_feat, H_feat, W_feat = feature_map.shape
        features = feature_map.flatten(2)  # [B*F, 384, 16]
        features = features.permute(0, 2, 1)  # [B*F, 16, 384]
        
        # Verify output shape
        if features.dim() != 3:
            raise ValueError(f"Expected 3D features [B*F, N, D], got: {features.shape}")
        
        BF_check, N, D = features.shape
        assert BF_check == B * F, f"Shape mismatch: {BF_check} != {B} * {F}"
        
        # Reshape to [B, F*N, D]
        features = features.reshape(B, F, N, D)
        features = features.reshape(B, F * N, D)
        
        return features
    
    def get_feature_info(self):
        """Return feature info"""
        return {
            'model_name': self.model_name,
            'embed_dim': self.embed_dim,
            'patch_size': self.patch_size,
            'architecture': 'EfficientViT-M5',
            'pretrain': 'GSViT',
        }


class GSViTWrapper(nn.Module):
    """
    GSViT model wrapper, handles color channel flip of input
    """
    def __init__(self, evit_model):
        super().__init__()
        self.evit = evit_model
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] Input image (RGB format)
        Returns:
            feature_map: [B, 384, 4, 4] Feature map
        """
        # GSViT requires color channel flip (RGB -> BGR)
        # Swap R and B channels
        x_flipped = x.clone()
        x_flipped[:, 0, :, :] = x[:, 2, :, :]  # R <- B
        x_flipped[:, 2, :, :] = x[:, 0, :, :]  # B <- R
        
        x = self.evit.patch_embed(x_flipped)
        x = self.evit.blocks1(x)
        x = self.evit.blocks2(x)
        x = self.evit.blocks3(x)
        return x  # [B, 384, 4, 4]


# Test code
if __name__ == "__main__":
    adapter = GSViTAdapter.from_config(
        resolution=224,
        checkpoint="ckpts/ckpts_foundation/GSViT.pkl",
        model_name='gsvit'
    )
    
    dummy_input = torch.randn(2, 3, 4, 224, 224)
    print(f"Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = adapter(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected: [2, {4 * 16}, {adapter.embed_dim}]")  # 4 frames * 16 patches = 64
    print(f"Feature info: {adapter.get_feature_info()}")