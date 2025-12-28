import torch
import torch.nn as nn
from typing import Optional
import sys
import os
sys.path.append(".")
from evals.foundation_phase_probing.modelcustom.adapters.base_adapter import BaseFoundationModelAdapter
from foundation_models.InternVideo.InternVideo2.multi_modality.models.backbones.internvideo2 import pretrain_internvideo2_1b_patch14_224
from foundation_models.InternVideo.InternVideo2.multi_modality.models.backbones.internvideo2.pos_embed import interpolate_pos_embed_internvideo2


class InternVideoAdapter(BaseFoundationModelAdapter):
    """InternVideo2 Adapter - Input format: [B, C, F, H, W]"""
    
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
            
        print(f"Loading {model_name} from {checkpoint}...")
        
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
                self.pretrained = None # We load manually
                self.drop_path_rate = 0.0 # Probing usually doesn't need drop path?
                
            def get(self, key, default=None):
                return getattr(self, key, default)

        class Config:
            def __init__(self):
                self.vision_encoder = VisionEncoderConfig()
                
        config = Config()
        
        # Build model
        # Note: We set pretrained=None in config to avoid auto loading, we do it manually to handle state_dict keys
        model = pretrain_internvideo2_1b_patch14_224(config)
        
        # Load weights
        if os.path.exists(checkpoint):
            print(f"Loading checkpoint from {checkpoint}")
            try:
                ckpt = torch.load(checkpoint, map_location='cpu')
                
                # Handle possible state_dict formats
                # - DeepSpeed checkpoints typically store weights under 'module'
                # - Some checkpoints store weights under 'model' or 'state_dict'
                if isinstance(ckpt, dict) and 'module' in ckpt and isinstance(ckpt['module'], dict):
                    state_dict = ckpt['module']
                    print("Detected DeepSpeed checkpoint format, using ckpt['module'] as state_dict.")
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
                # We need keys for the vision encoder only.
                # Usually prefixed with 'vision_encoder.' in Stage 2 checkpoints
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
                        print("Detected multimodal checkpoint keys, extracted ONLY vision encoder weights (prefix 'vision_encoder.').")
                
                # Final safety: keep only keys that exist in the current model to avoid silently carrying unrelated tensors.
                if isinstance(state_dict, dict):
                    model_keys = set(model.state_dict().keys())
                    before_cnt = len(state_dict)
                    state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
                    after_cnt = len(state_dict)
                    if after_cnt != before_cnt:
                        print(f"Filtered state_dict by model keys: {before_cnt} -> {after_cnt}")
                
                # Interpolate position embeddings if resolution/frames differ
                # model expects orig_t_size=8 by default in code, but our model might be f4.
                # If checkpoint is f4, we should probably tell interpolation that origin is 4?
                # However, pretrain_internvideo2_1b_patch14_224 defaults to orig_t_size=8 in its load logic.
                # We should call interpolate_pos_embed_internvideo2 manually.
                
                # The checkpoint provided is f4, so orig_t_size should be 4.
                # If we change num_frames in config, we need interpolation.
                try:
                    interpolate_pos_embed_internvideo2(state_dict, model, orig_t_size=4) 
                except Exception as e:
                    print(f"Pos embed interpolation warning: {e}")

                msg = model.load_state_dict(state_dict, strict=False)
                print(f"Weights loaded: {msg}")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
        else:
            print(f"Checkpoint not found at {checkpoint}, using random init (WARNING)")
            
        embed_dim = model.embed_dim # 1408
        
        return cls(model, embed_dim, model_name, resolution, num_frames)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, F, H, W]
        Returns:
            features: [B, F*N, D]
        """
        B, C, F, H, W = x.shape
        
        # Resize if needed
        if H != self.resolution or W != self.resolution:
            # [B, C, F, H, W] -> [B*F, C, H, W]
            x = x.permute(0, 2, 1, 3, 4).reshape(B*F, C, H, W)
            x = torch.nn.functional.interpolate(
                x, size=(self.resolution, self.resolution), 
                mode='bicubic', align_corners=False
            )
            # -> [B, F, C, H, W] -> [B, C, F, H, W]
            x = x.reshape(B, F, C, self.resolution, self.resolution).permute(0, 2, 1, 3, 4)
            
        # InternVideo2 forward expects [B, C, T, H, W] which is what we have (F=T)
        # But we must ensure F matches self.num_frames or handle it?
        # The model's pos_embed is fixed size.
        # If F != self.num_frames, the model forward will fail or produce wrong results due to shape mismatch in pos_embed addition.
        # For now, we assume F matches or we let it fail/warn.
        if F != self.num_frames:
            # Ideally we should interpolate pos_embed here dynamically, but InternVideo2 class doesn't support dynamic pos_embed easily in forward.
            # We can try to rely on the fact that if we initialized with F frames, it expects F frames.
            pass
            
        # Forward pass
        # x_vis_only=True returns x_vis [B, T*L+1, C]
        features = self.model(x, x_vis_only=True)
        
        # Remove CLS token (first token)
        features = features[:, 1:, :] # [B, T*L, C]
        
        # Verify shape
        # [B, F*N, D]
        # InternVideo2 flattens T and L together.
        # L = (H/P) * (W/P)
        # So T*L is F*N.
        
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
