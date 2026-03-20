"""
EndoMamba Foundation Model Adapter
Based on official EndoMamba repo definition, supports loading EndoMamba checkpoint
"""

import sys
sys.path.append(".")

import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from pathlib import Path
from functools import partial

from evals.foundation_phase_probing.modelcustom.adapters.base_adapter import BaseFoundationModelAdapter
from evals.foundation_phase_probing.modelcustom.adapters.utils import load_checkpoint_generic


def interpolate_pos_embed(pos_embed: torch.Tensor, target_num_patches: int) -> torch.Tensor:
    """
    Interpolate position encoding to match target patch count
    
    Args:
        pos_embed: [1, N+1, D] Original position encoding (includes CLS token)
        target_num_patches: Target patch count (excluding CLS)
    
    Returns:
        Interpolated position encoding [1, target_num_patches+1, D]
    """
    # Separate CLS token and patch tokens
    cls_token = pos_embed[:, :1, :]  # [1, 1, D]
    patch_pos_embed = pos_embed[:, 1:, :]  # [1, N, D]
    
    N = patch_pos_embed.shape[1]
    if N == target_num_patches:
        return pos_embed
    
    # Compute source and target spatial dimensions
    src_size = int(N ** 0.5)
    tgt_size = int(target_num_patches ** 0.5)
    
    logger.debug("Interpolating pos_embed: %sx%s -> %sx%s", src_size, src_size, tgt_size, tgt_size)
    
    # Reshape to 2D spatial format for interpolation
    # [1, N, D] -> [1, D, src_size, src_size]
    D = patch_pos_embed.shape[-1]
    patch_pos_embed = patch_pos_embed.reshape(1, src_size, src_size, D).permute(0, 3, 1, 2)
    
    # Bicubic interpolation
    patch_pos_embed = F.interpolate(
        patch_pos_embed,
        size=(tgt_size, tgt_size),
        mode='bicubic',
        align_corners=False
    )
    
    # [1, D, tgt_size, tgt_size] -> [1, target_num_patches, D]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, target_num_patches, D)
    
    # Reconcatenate CLS token
    return torch.cat([cls_token, patch_pos_embed], dim=1)


class EndoMambaAdapter(BaseFoundationModelAdapter):
    """
    Adapter for EndoMamba video model.

    Input format: [B, C, F, H, W]. Model consumes 5D directly (F = temporal). Optional resize to 224×224.
    Output: [B, F*N, D] (patch tokens from all frames, CLS removed).
    """
    
    # Original image size used when training the checkpoint
    CHECKPOINT_IMG_SIZE = 224
    PATCH_SIZE = 16
    
    def __init__(self, model, embed_dim: int, model_name: str):
        super().__init__(model, embed_dim)
        self.model_name = model_name
    
    @classmethod
    def from_config(cls, resolution: int, checkpoint: Optional[str] = None, model_name: str = 'endomamba_small'):
        """
        Create adapter from config, using official EndoMamba definition
        
        Args:
            resolution: Input resolution (typically 224)
            checkpoint: Checkpoint path
            model_name: Model architecture name ('endomamba_tiny', 'endomamba_small', 'endomamba_middle')
        """
        import sys
        
        # Add EndoMamba path to sys.path
        endomamba_path = Path(__file__).parent.parent.parent.parent.parent / "foundation_models" / "EndoMamba" / "videomamba"
        if str(endomamba_path) not in sys.path:
            sys.path.insert(0, str(endomamba_path))
        # Import official model definition
        from video_sm.models.endomamba import endomamba_small
        
        logger.info("Loading EndoMamba model: %s", model_name)
        
        try:
            # Select model architecture based on model_name
            model_factory = {
                'endomamba_small': endomamba_small
            }
            
            if model_name not in model_factory:
                raise ValueError(f"Unknown model_name: {model_name}. Choose from {list(model_factory.keys())}")
            
            # Get embed_dim
            embed_dim_map = {
                'endomamba_small': 384,
            }
            embed_dim = embed_dim_map.get(model_name, 384)
            patch_size = cls.PATCH_SIZE
            
            # Create model using checkpoint's original image size (224)
            model = model_factory[model_name](
                pretrained=False, 
                img_size=cls.CHECKPOINT_IMG_SIZE,  # Use size from checkpoint training
                num_classes=0,
                with_head=False
            )
            
            logger.info("EndoMamba model created: embed_dim=%s patch_size=%s img_size=%s", embed_dim, patch_size, cls.CHECKPOINT_IMG_SIZE)
            
            # Load checkpoint
            default_path = "ckpts/ckpts_foundation/endomamba_checkpoint-best.pth"
            ckpt_path = checkpoint if checkpoint else default_path
            
            ckpt, _ = load_checkpoint_generic(ckpt_path, verbose=True)
            
            if ckpt is not None:
                # Get state_dict
                if 'model' in ckpt:
                    state_dict = ckpt['model']
                elif 'state_dict' in ckpt:
                    state_dict = ckpt['state_dict']
                else:
                    state_dict = ckpt
                
                # Check if pos_embed interpolation is needed
                model_num_patches = model.patch_embed.num_patches
                if 'pos_embed' in state_dict:
                    ckpt_pos_embed = state_dict['pos_embed']
                    ckpt_num_patches = ckpt_pos_embed.shape[1] - 1  # Subtract CLS token
                    
                    if ckpt_num_patches != model_num_patches:
                        logger.debug("pos_embed mismatch: checkpoint has %s patches, model expects %s", ckpt_num_patches, model_num_patches)
                        state_dict['pos_embed'] = interpolate_pos_embed(ckpt_pos_embed, model_num_patches)
                
                # Load state_dict (strict=False to ignore head and other unneeded params)
                msg = model.load_state_dict(state_dict, strict=False)
                logger.info("Checkpoint loaded: missing_keys=%s unexpected_keys=%s", len(msg.missing_keys), len(msg.unexpected_keys))
                if msg.unexpected_keys:
                    logger.debug("Unexpected keys (ignored): %s", msg.unexpected_keys[:5])
            else:
                logger.warning("Could not load checkpoint from %s", ckpt_path)

        except Exception as e:
            logger.exception("Error loading EndoMamba model: %s", e)
            raise e
        
        return cls(model, embed_dim, model_name)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, F, H, W] Video input
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

        # EndoMamba expects input format [B, C, T, H, W] where T is temporal dimension
        # Our input is [B, C, F, H, W], so F corresponds to T
        # Use directly since dimension order is the same
        
        # Resize if dimensions do not match (EndoMamba typically uses 224x224)
        target_H, target_W = 224, 224
        if H != target_H or W != target_W:
            import torch.nn.functional as fn
            # [B, C, F, H, W] → [B*F, C, H, W] → resize → [B*F, C, 224, 224]
            x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
            x_resized = fn.interpolate(x_reshaped, size=(target_H, target_W), 
                                       mode='bilinear', align_corners=False)
            # [B*F, C, 224, 224] → [B, F, C, 224, 224] → [B, C, F, 224, 224]
            x = x_resized.reshape(B, F, C, target_H, target_W).permute(0, 2, 1, 3, 4)
            H, W = target_H, target_W
        
        # EndoMamba's forward_features expects input [B, C, T, H, W], Our input is already [B, C, F, H, W], F corresponds to T
        # EndoMamba's forward_features returns (hidden_states, inference_params)
        with torch.no_grad():
            # forward_features returns features in format [B, T, N+1, D]
            # where N+1 includes CLS token (position 0) and N patch tokens
            features, _ = self.model.forward_features(x, inference_params=None)
        
        # Handle different output formats
        if features.dim() == 4:
            # Output format: [B, T, N+1, D] - includes CLS token and patch tokens per frame
            B_out, T, N_plus_1, D = features.shape
            assert B_out == B, f"Batch size mismatch: {B_out} != {B}"
            assert T == F, f"Frame count mismatch: {T} != {F}"
            
            # Remove CLS token (position 0) per frame, keep only patch tokens
            # [B, T, N+1, D] -> [B, T, N, D]
            patch_tokens = features[:, :, 1:, :]  # Skip CLS token
            
            # Concatenate patch tokens from all frames into [B, T*N, D]
            # [B, T, N, D] -> [B, T*N, D]
            N = N_plus_1 - 1  # Patch count (excluding CLS)
            features = patch_tokens.reshape(B, T * N, D)
            
        elif features.dim() == 3:
            # Output format: [B, T*N, D] - already flattened
            B_out, TN, D = features.shape
            assert B_out == B, f"Batch size mismatch: {B_out} != {B}"
            
        elif features.dim() == 2:
            # Output format: [B, D] - globally pooled features
            # Need to expand dim to match expected [B, N, D] format
            B_out, D = features.shape
            features = features.unsqueeze(1)  # [B, 1, D]
            
        else:
            raise ValueError(f"Unexpected features shape: {features.shape}")
        
        assert features.shape[-1] == self.embed_dim, f"Embed dim mismatch: {features.shape[-1]} != {self.embed_dim}"
        
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
    """
    Input shape: torch.Size([2, 3, 4, 224, 224])
    Output shape: torch.Size([2, 784, 384])  # For small model
    Expected: [2, 4*196, 384] = [2, 784, 384]
    """
    # Ensure GPU is used
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test input/output format
    adapter = EndoMambaAdapter.from_config(
        resolution=224,
        checkpoint="/home/chen_chuxi/NSJepa/ckpts_foundation/endomamba_checkpoint-best.pth",
        model_name='endomamba_small'
    )
    adapter = adapter.to(device)  # Move model to GPU
    
    # Simulate input: [B=2, C=3, F=4, H=224, W=224]
    dummy_input = torch.randn(2, 3, 4, 224, 224).to(device)  # Move input to GPU
    
    print(f"Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = adapter(dummy_input)
    
    print(f"Output shape: {output.shape}")  # Should be [2, 4*N, D]
    num_patches_per_frame = (224 // 16) ** 2  # 196 patches per frame
    print(f"Expected: [2, {4 * num_patches_per_frame}, {adapter.embed_dim}]")
    print(f"Feature info: {adapter.get_feature_info()}")
    print("✅ EndoMamba adapter test passed!")