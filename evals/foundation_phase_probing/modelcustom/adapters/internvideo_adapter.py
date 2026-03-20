"""
InternVideo2 Foundation Model Adapter
Based on official InternVideo2 repo definition, supports loading InternVideo2 checkpoint
"""

import os
import sys
sys.path.append(".")

import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from typing import Optional

from evals.foundation_phase_probing.modelcustom.adapters.base_adapter import BaseFoundationModelAdapter
from foundation_models.InternVideo.InternVideo2.multi_modality.models.backbones.internvideo2 import pretrain_internvideo2_1b_patch14_224
from foundation_models.InternVideo.InternVideo2.multi_modality.models.backbones.internvideo2.pos_embed import interpolate_pos_embed_internvideo2


class InternVideoAdapter(BaseFoundationModelAdapter):
    """
    Adapter for InternVideo2 video backbone.

    Input format: [B, C, F, H, W]. Model consumes 5D directly (F must match num_frames).
    Output: [B, F*N, D] (patch tokens, CLS removed). Optional resize to resolution×resolution.
    """
    
    def __init__(self, model, embed_dim: int, model_name: str, resolution: int = 224, num_frames: int = 4):
        super().__init__(model, embed_dim)
        self.model_name = model_name
        self.resolution = resolution
        self.num_frames = num_frames
        
    @classmethod
    def from_config(cls, resolution: int = 224, checkpoint: Optional[str] = None, model_name: str = 'InternVideo2-Stage2_1B-224p-f4', num_frames: int = 4):
        """
        Create adapter from config
        """
        # Default checkpoint path if not provided
        if checkpoint is None:
            checkpoint = "ckpts/ckpts_foundation/OpenGVLab/InternVideo2-Stage2_1B-224p-f4/InternVideo2-stage2_1b-224p-f4.pt"
            
        logger.info("Loading %s from %s", model_name, checkpoint)
        
        # Mock Config object required by InternVideo2 builder
        class VisionEncoderConfig:
            def __init__(self):
                self.name = 'pretrain_internvideo2_1b_patch14_224'
                self.img_size = resolution
                self.patch_size = 14
                self.num_frames = num_frames
                self.tubelet_size = 1
                self.clip_embed_dim = 768 # Default for 1B
                self.clip_teacher_embed_dim = 3200
                self.clip_teacher_final_dim = 768
                self.clip_norm_type = 'l2'
                self.clip_return_layer = 1
                self.clip_student_return_interval = 1
                self.sep_image_video_pos_embed = False
                self.use_checkpoint = False # set True to save memory if needed
                self.checkpoint_num = 0
                self.pretrained = None # load manually
                self.drop_path_rate = 0.0
                
            def get(self, key, default=None):
                return getattr(self, key, default)

        class Config:
            def __init__(self):
                self.vision_encoder = VisionEncoderConfig()
                
        config = Config()
        
        # Build model, set pretrained=None in config to avoid auto loading, we do it manually to handle state_dict keys
        model = pretrain_internvideo2_1b_patch14_224(config)
        
        # Load weights
        if os.path.exists(checkpoint):
            logger.info("Loading checkpoint from %s", checkpoint)
            try:
                ckpt = torch.load(checkpoint, map_location='cpu')
                
                # Handle possible state_dict formats
                # - DeepSpeed checkpoints typically store weights under 'module'
                # - Some checkpoints store weights under 'model' or 'state_dict'
                if isinstance(ckpt, dict) and 'module' in ckpt and isinstance(ckpt['module'], dict):
                    state_dict = ckpt['module']
                    logger.info("Detected DeepSpeed checkpoint format, using ckpt['module'] as state_dict")
                elif isinstance(ckpt, dict) and 'model' in ckpt and isinstance(ckpt['model'], dict):
                    state_dict = ckpt['model']
                elif isinstance(ckpt, dict) and 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
                    state_dict = ckpt['state_dict']
                else:
                    state_dict = ckpt
                
                # Some checkpoints may prefix keys with 'module.' (DDP). Strip it if present.
                if isinstance(state_dict, dict) and any(k.startswith('module.') for k in state_dict.keys()):
                    state_dict = {k[len('module.'):]: v for k, v in state_dict.items() if k.startswith('module.')}
                    
                # Filter and rename keys if they are from Stage 2 (multimodal) checkpoint
                if isinstance(state_dict, dict):
                    has_vision_prefix = any(k.startswith('vision_encoder.') for k in state_dict.keys())
                    if has_vision_prefix:
                        # IMPORTANT: when we detect a multimodal checkpoint, ONLY keep vision_encoder weights.
                        # Otherwise we may silently pass unrelated keys (and miss loading the actual vision weights).
                        state_dict = {
                            k[len('vision_encoder.'):]: v
                            for k, v in state_dict.items()
                            if k.startswith('vision_encoder.')
                        }
                        logger.info("Detected multimodal checkpoint keys, extracted ONLY vision encoder weights (prefix 'vision_encoder.')")
                
                # Final safety: keep only keys that exist in the current model to avoid silently carrying unrelated tensors.
                if isinstance(state_dict, dict):
                    model_keys = set(model.state_dict().keys())
                    before_cnt = len(state_dict)
                    state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
                    after_cnt = len(state_dict)
                    if after_cnt != before_cnt:
                        logger.debug("Filtered state_dict by model keys: %s -> %s", before_cnt, after_cnt)
                
                # Interpolate position embeddings if resolution/frames differ
                try:
                    interpolate_pos_embed_internvideo2(state_dict, model, orig_t_size=4) 
                except Exception as e:
                    logger.warning("Pos embed interpolation: %s", e)

                msg = model.load_state_dict(state_dict, strict=False)
                logger.info("Weights loaded: %s", msg)
            except Exception as e:
                logger.exception("Error loading checkpoint: %s", e)
        else:
            logger.warning("Checkpoint not found at %s, using random init", checkpoint)
            
        embed_dim = model.embed_dim # 1408
        
        return cls(model, embed_dim, model_name, resolution, num_frames)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, F, H, W] video input
               - B: batch size, C: channels (3), F: frames (should match num_frames), H, W: height and width

        Returns:
            features: [B, F*N, D]
               - F*N: patch tokens from all frames (CLS removed); D: embed_dim
        """
        B, C, F, H, W = x.shape
        
        if H != self.resolution or W != self.resolution:
            # [B, C, F, H, W] → [B, F, C, H, W] → [B*F, C, H, W] → resize → [B, C, F, H', W']
            x = x.permute(0, 2, 1, 3, 4).reshape(B*F, C, H, W)
            x = torch.nn.functional.interpolate(
                x, size=(self.resolution, self.resolution), 
                mode='bicubic', align_corners=False
            )
            # -> [B, F, C, H, W] -> [B, C, F, H, W]
            x = x.reshape(B, F, C, self.resolution, self.resolution).permute(0, 2, 1, 3, 4)
            
        # Model expects [B, C, T, H, W] (F=T). F should match self.num_frames for pos_embed.
        features = self.model(x, x_vis_only=True)  # [B, T*L+1, C]
        features = features[:, 1:, :]  # Remove CLS -> [B, F*N, D]
        return features

    def get_feature_info(self):
        return {
            'model_name': self.model_name,
            'embed_dim': self.embed_dim,
            'resolution': self.resolution,
            'num_frames': self.num_frames
        }

if __name__ == "__main__":
    # Test
    try:
        adapter = InternVideoAdapter.from_config(
            resolution=224, 
            num_frames=4,
            checkpoint="ckpts/ckpts_foundation/OpenGVLab/InternVideo2-Stage2_1B-224p-f4/InternVideo2-stage2_1b-224p-f4.pt"
        )
        dummy_input = torch.randn(2, 3, 4, 224, 224)
        output = adapter(dummy_input)
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"Test failed: {e}")
