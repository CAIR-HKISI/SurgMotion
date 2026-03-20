"""
SurgVISTA Foundation Model Adapter
Based on the official SurgVISTA repository definition, supports loading SurgVISTA checkpoint.
Follows the design of the EndoViT adapter.
"""

import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from typing import Optional
from pathlib import Path
from functools import partial

from evals.foundation_phase_probing.modelcustom.adapters.base_adapter import BaseFoundationModelAdapter
from evals.foundation_phase_probing.modelcustom.adapters.utils import load_and_apply_checkpoint


class SurgVISTAAdapter(BaseFoundationModelAdapter):
    """
    Adapter for SurgVISTA video ViT.

    Input format: [B, C, F, H, W]. Optional resize to 224×224. Model patch_embed consumes 5D;
    output is [B, N, D] with N = all patch tokens from all frames.
    """
    
    def __init__(self, model, embed_dim: int, model_name: str):
        super().__init__(model, embed_dim)
        self.model_name = model_name
    
    @classmethod
    def from_config(cls, resolution: int, checkpoint: Optional[str] = None, model_name: str = 'vit_base_patch16'):
        """
        Create adapter from config, using official SurgVISTA definition
        
        Args:
            resolution: Input resolution (typically 224)
            checkpoint: Checkpoint path, defaults to SurgVISTA pretrained weights
            model_name: Model architecture name ('vit_base_patch16', 'vit_large_patch16')
        """
        import sys
        
        # Add SurgVISTA path to sys.path
        surgvista_path = Path(__file__).parent.parent.parent.parent.parent / "foundation_models" / "SurgVISTA" / "downstream"
        if str(surgvista_path) not in sys.path:
            sys.path.insert(0, str(surgvista_path))
        
        # Import official model definition and utilities
        from model.unifiedmodel import unified_base
        import utils
        
        logger.info("Loading SurgVISTA model: %s", model_name)
        
        try:
            # Create model (no mean pooling, keep all tokens)
            model = unified_base(
                pretrained=False,
                pretrain_path=None,
                num_classes=0,  # No classification head needed
                use_mean_pooling=False,  # Key: keep all tokens
                all_frames=128,
                tubelet_size=2,
            )
            
            embed_dim = 768
            patch_size = 16
            
            logger.info("SurgVISTA model created: embed_dim=%s patch_size=%s", embed_dim, patch_size)
            
            # Load checkpoint
            success, info = load_and_apply_checkpoint(
                model=model,
                checkpoint_path=checkpoint,
                default_path="/home/chen_chuxi/NSJepa/ckpts_foundation/vit_large_patch16_224_surgery.pth",
                strict=False,
                key_prefix_to_remove=None,
                verbose=True
            )
            
            if not success:
                logger.warning("%s", info)

        except Exception as e:
            logger.exception("Error loading SurgVISTA model: %s", e)
            raise e
        
        return cls(model, embed_dim, model_name)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, F, H, W] video input
               - B: batch size, C: channels (3), F: frames, H, W: height and width

        Returns:
            features: [B, N, D]
               - N: all patch tokens from all frames (model-internal flattening); D: embed_dim
        """
        B, C, F, H, W = x.shape

        # Resize to 224x224
        target_H, target_W = 224, 224
        if H != target_H or W != target_W:
            import torch.nn.functional as fn
            x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
            x_resized = fn.interpolate(x_reshaped, size=(target_H, target_W), 
                                       mode='bilinear', align_corners=False)
            x = x_resized.reshape(B, F, C, target_H, target_W).permute(0, 2, 1, 3, 4)
            H, W = target_H, target_W
        
        # Extract all patch tokens (patch_embed accepts [B, C, F, H, W]; model flattens internally)
        with torch.no_grad():
            # Patch embedding
            x_tokens = self.model.patch_embed(x)  # [B, N, D]
            
            # Position embedding
            if self.model.pos_embed is not None:
                B_tokens, N_tokens, D_tokens = x_tokens.shape
                if isinstance(self.model.pos_embed, nn.Parameter):
                    # Learnable position encoding: use directly
                    x_tokens = x_tokens + self.model.pos_embed.expand(B_tokens, -1, -1).type_as(x_tokens).to(x_tokens.device).clone().detach()
                else:
                    # Sinusoidal position encoding: need to handle frame count mismatch
                    # If input frame count doesn't match position encoding, interpolate
                    pos_embed = self.model.pos_embed
                    if isinstance(pos_embed, torch.Tensor):
                        pos_embed_tensor = pos_embed
                    else:
                        # numpy array
                        pos_embed_tensor = torch.from_numpy(pos_embed).to(x_tokens.device).type_as(x_tokens)
                    
                    # Check if position encoding length matches
                    if pos_embed_tensor.shape[1] != N_tokens:
                        # Position encoding length mismatch, interpolate
                        # pos_embed: [1, N_old, D] -> [1, N_new, D]
                        import torch.nn.functional as F
                        pos_embed_tensor = F.interpolate(
                            pos_embed_tensor.transpose(1, 2),  # [1, D, N_old]
                            size=N_tokens,  # Interpolate to N_tokens
                            mode='linear',
                            align_corners=False
                        ).transpose(1, 2)  # [1, N_new, D]
                    
                    x_tokens = x_tokens + pos_embed_tensor.expand(B_tokens, -1, -1)
            
            x_tokens = self.model.pos_drop(x_tokens)
            
            # Transformer blocks
            for blk in self.model.blocks:
                x_tokens = blk(x_tokens)
            
            # Final norm
            features = self.model.norm(x_tokens)  # [B, N, D]
        
        return features  # [B, N, D], N = all patches from all frames
    
    def get_feature_info(self):
        """Return feature extraction info"""
        return {
            'model_name': self.model_name,
            'embed_dim': self.embed_dim,
            'patch_size': 16,
            'tubelet_size': getattr(self.model.patch_embed, 'tubelet_size', 2),
            'max_frames': 128,  # Keep consistent with other models
        }


if __name__ == "__main__":
    adapter = SurgVISTAAdapter.from_config(
        resolution=224,
        checkpoint="/home/chen_chuxi/NSJepa/ckpts_foundation/vit_large_patch16_224_surgery.pth",
        model_name='vit_base_patch16'
    )
    
    dummy_input = torch.randn(2, 3, 16, 224, 224)
    print(f"Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = adapter(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Feature info: {adapter.get_feature_info()}")