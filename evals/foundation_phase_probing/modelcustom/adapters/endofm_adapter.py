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
from evals.foundation_phase_probing.modelcustom.adapters.utils import load_and_apply_checkpoint, parse_args, load_config


class EndoFMAdapter(BaseFoundationModelAdapter):
    """
    Adapter for Endo-FM (TimeSformer-style) video model.

    Input format: [B, C, F, H, W]. The model consumes 5D video directly (no frame flattening to B*F).
    Output format: [B, N, D] where N is total patch tokens from the model (CLS removed).
    """
    
    def __init__(self, model, embed_dim: int, model_name: str):
        super().__init__(model, embed_dim)
        self.model_name = model_name
        self.patch_size = 16
    
    @classmethod
    def from_config(cls, resolution: int, frames_per_clip: int, checkpoint: Optional[str] = None, model_name: str = ''):
        """
        Create adapter from config, using official EndoViT definition.

        Args:
            resolution: Input resolution (typically 224)
            checkpoint: Checkpoint path, default is EndoViT_SPR
            model_name: Model architecture name ('vit_base_patch16', 'vit_large_patch16', 'vit_huge_patch14')
        """
        import sys
        
        # Add EndoFM path to sys.path
        endofm_path = Path(__file__).parent.parent.parent.parent.parent / "foundation_models" / "Endo-FM"
        if str(endofm_path) not in sys.path:
            sys.path.insert(0, str(endofm_path))
        
        # Import official model definition and config utilities
        from models import get_vit_base_patch16_224

        logger.info("Loading Endo-FM model: %s", model_name)
        
        try:
            # Create config object (simulate argparse)
            class SimpleArgs:
                def __init__(self):
                    self.cfg_file = str(endofm_path / "models" / "configs" / "Kinetics" / "TimeSformer_divST_8x32_224.yaml")
                    self.opts = None
                    self.num_shards = 1
                    self.shard_id = 0
            
            # Load Endo-FM config
            args = SimpleArgs()
            config = load_config(args)
            
            # Modify config to fit our setup
            config.DATA.TRAIN_CROP_SIZE = resolution
            config.DATA.NUM_FRAMES = frames_per_clip
            config.MODEL.NUM_CLASSES = 400  # Default value, head will be removed later
            config.TIMESFORMER.ATTENTION_TYPE = 'divided_space_time'  # Endo-FM standard attention type

            # Create model (no_head=True means no classification head)
            model = get_vit_base_patch16_224(cfg=config, no_head=True)
            embed_dim = 768  # ViT-Base embed_dim
            patch_size = 16
            
            logger.info(
                "Endo-FM created: embed_dim=%s patch_size=%s attention_type=%s num_frames=%s resolution=%s",
                embed_dim, patch_size, config.TIMESFORMER.ATTENTION_TYPE, frames_per_clip, resolution
            )
            
            # Use common utility to load checkpoint
            success, info = load_and_apply_checkpoint(
                model=model,
                checkpoint_path=checkpoint,
                default_path="ckpts/ckpts_foundation/endo_fm.pth",
                strict=False,
                key_prefix_to_remove="backbone.",  # Endo-FM weights have 'backbone.' prefix
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
               - B: batch size, C: channels (3), F: frames, H, W: height and width

        Returns:
            features: [B, N, D]
               - N: total patch tokens (model-internal); D: embed_dim
        """
        B, C, F, H, W = x.shape

        # Extract features via Endo-FM
        with torch.no_grad():
            # Use official forward_features method
            # pool_type='no_pooling' returns [B*F, N, D] (excluding CLS token)
            features = self.model.forward_features(x, get_all=True)
        
        # Verify output shape
        if features.dim() != 3:
            raise ValueError(f"Expected 3D features [B*F, N, D], got shape: {features.shape}")
        features = features[:, 1:, :]  # Remove CLS token -> [B, N, D]

        B, N, D = features.shape
        return features

    def get_feature_info(self):
        """Return feature extraction info (for debugging)"""
        return {
            'model_name': self.model_name,
            'embed_dim': self.embed_dim,
            'patch_size': 16,
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
    adapter = EndoFMAdapter.from_config(
        resolution=256,
        frames_per_clip=64,
        checkpoint="ckpts/ckpts_foundation/endofm_cholec80.pth",
        model_name='endofm'
    )
    
    # Simulated input: [B=2, C=3, F=4, H=224, W=224]
    dummy_input = torch.randn(2, 3, 4, 224, 224)
    
    print(f"Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = adapter(dummy_input)
    
    print(f"Output shape: {output.shape}")  # Should be [2, 4*N, D]
    print(f"Expected: [2, {4 * (224//16)**2}, {adapter.embed_dim}]")
    print(f"Feature info: {adapter.get_feature_info()}")