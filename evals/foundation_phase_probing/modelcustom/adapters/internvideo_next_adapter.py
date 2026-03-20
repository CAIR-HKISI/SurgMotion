"""
InternVideo2-Chat (Next) Adapter
Refactored for attention probing
"""

import sys
sys.path.append(".")

import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Optional

from evals.foundation_phase_probing.modelcustom.adapters.base_adapter import BaseFoundationModelAdapter


class InternVideoNextAdapter(BaseFoundationModelAdapter):
    """
    Adapter for InternVideo2-Chat vision backbone on video input.

    Input format: [B, C, F, H, W]. Model consumes 5D directly. Output: [B, F*N, D] (CLS removed if present).
    Optional resize to resolution×resolution.
    """
    
    def __init__(self, model, embed_dim: int, model_name: str, resolution: int = 224):
        super().__init__(model, embed_dim)
        self.model_name = model_name
        self.resolution = resolution
        
    @classmethod
    def from_config(
        cls,
        resolution: int,
        checkpoint: Optional[str] = None,
        model_name: str = "internvideo_next_large_p14_res224_f16",
    ):
        """
        Create adapter from config (style aligned with dinov3_adapter)
        """
        import traceback

        # Default checkpoint path mapping
        MODEL_PATHS = {
            "internvideo_next_large_p14_res224_f16": "ckpts/ckpts_foundation/revliter/internvideo_next_large_p14_res224_f16",
        }

        try:
            if checkpoint:
                pretrained_model_name = checkpoint
            else:
                pretrained_model_name = MODEL_PATHS.get(model_name, model_name)

            logger.info("Loading %s from %s", model_name, pretrained_model_name)
            
            # --- Monkeypatch to handle flash_attn issues safely ---
            import sys
            import types
            
            class DummyModule:
                __path__ = []
                def __init__(self):
                    self.__spec__ = None
                    self.__loader__ = None
                    self.__package__ = None
                    self.__name__ = "flash_attn"
                    
                def __getattr__(self, name):
                    if name in ["__spec__", "__loader__", "__package__", "__name__", "__path__"]:
                        return None
                    if name in ["flash_attn_interface", "bert_padding", "modules", "ops"]:
                        return self
                    # Raising ImportError on attribute access usually simulates "from module import name" failure
                    raise ImportError(f"Dummy module: {name} not found")
            
            # Helper to inject dummy modules if they don't exist
            def ensure_dummy_module(name):
                if name not in sys.modules:
                    sys.modules[name] = DummyModule()
            
            # Only block if not already imported successfully
            if 'flash_attn' not in sys.modules:
                 ensure_dummy_module('flash_attn')
                 ensure_dummy_module('flash_attn.flash_attn_interface')
                 ensure_dummy_module('flash_attn.bert_padding')
                 ensure_dummy_module('flash_attn.modules')
                 ensure_dummy_module('flash_attn.modules.mlp')
                 ensure_dummy_module('flash_attn.ops')
                 ensure_dummy_module('flash_attn.ops.rms_norm')

            # Here we force-disable the fused path via config.model_config to ensure usability.
            config = AutoConfig.from_pretrained(pretrained_model_name, trust_remote_code=True)
            if hasattr(config, "model_config") and isinstance(config.model_config, dict):
                config.model_config["use_flash_attn"] = False
                config.model_config["use_fused_rmsnorm"] = False
                config.model_config["use_fused_mlp"] = False

            # Wrapper will initialize on CPU first, then uniformly .to(device)
            model = AutoModel.from_pretrained(
                pretrained_model_name,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map=None,
                config=config,
            )
            
            # Monkeypatch loaded model to use SDPA (PyTorch 2.0+) for speed
            try:
                def efficient_attn_forward(self, x):
                    B, N, C = x.shape
                    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv.unbind(0)
                    
                    if self.qk_normalization:
                        B_, H_, N_, D_ = q.shape
                        q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
                        k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
                    
                    # Use PyTorch SDPA
                    # q, k, v are [B, H, N, D]
                    x = F.scaled_dot_product_attention(
                        q, k, v,
                        dropout_p=self.attn_drop.p if self.training else 0.0,
                        scale=self.scale
                    )
                    
                    x = x.transpose(1, 2).reshape(B, N, C)
                    x = self.proj(x)
                    x = self.proj_drop(x)
                    return x

                # Apply patch to all Attention modules in the backbone
                # The model structure is usually model.model.blocks...
                vision_encoder = None
                if hasattr(model, "vision_model"):
                    vision_encoder = model.vision_model
                elif hasattr(model, "model") and hasattr(model.model, "patch_embed"):
                    vision_encoder = model.model
                elif hasattr(model, "patch_embed"):
                    vision_encoder = model
                
                if vision_encoder and hasattr(vision_encoder, "blocks"):
                    logger.debug("Applying efficient SDPA patch to InternVideoNext Attention blocks")
                    count = 0
                    for block in vision_encoder.blocks:
                        if hasattr(block, "attn"):
                            # Bind the method to the instance (or just replace the method on the class if consistent)
                            # Replacing on instance is safer
                            block.attn._naive_attn = types.MethodType(efficient_attn_forward, block.attn)
                            # Force use_flash_attn=False so it calls _naive_attn (which is now patched)
                            block.attn.use_flash_attn = False
                            count += 1
                    logger.debug("Patched %s attention blocks with SDPA", count)
            except Exception as patch_e:
                logger.warning("Failed to apply SDPA patch: %s", patch_e)
            
            model.eval()

            # Extract Vision Encoder (InternVideoNext remote code: model.model is the backbone)
            if hasattr(model, "vision_model"):
                vision_encoder = model.vision_model
            elif hasattr(model, "model") and hasattr(model.model, "patch_embed"):
                vision_encoder = model.model
            elif hasattr(model, "patch_embed"):
                vision_encoder = model
            else:
                raise ValueError(f"Could not find vision encoder in loaded model. attrs={dir(model)}")

            vision_encoder.eval()

            # Auto-detect embed_dim
            if hasattr(vision_encoder, "embed_dim"):
                embed_dim = int(vision_encoder.embed_dim)
            elif hasattr(vision_encoder, "config") and hasattr(vision_encoder.config, "hidden_size"):
                embed_dim = int(vision_encoder.config.hidden_size)
            else:
                raise ValueError("Could not detect embed_dim from InternVideoNext vision encoder.")

            logger.info("InternVideoNext loaded: embed_dim=%s", embed_dim)

            return cls(vision_encoder, embed_dim, model_name, resolution)

        except Exception as e:
            logger.exception("Error loading InternVideoNext model: %s", e)
            raise
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, F, H, W] video input
               - B: batch size, C: channels (3), F: frames, H, W: height and width

        Returns:
            features: [B, F*N, D]
               - F*N: patch tokens from all frames (CLS removed if present); D: embed_dim
        """
        B, C, T, H, W = x.shape
        
        target_H, target_W = self.resolution, self.resolution
        if H != target_H or W != target_W:
            # [B, C, F, H, W] → [B, F, C, H, W] → [B*F, C, H, W] → resize → [B, C, F, H', W']
            x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            x_resized = F.interpolate(
                x_reshaped, 
                size=(target_H, target_W), 
                mode='bicubic', 
                align_corners=False
            )
            # -> [B, T, C, H', W'] -> [B, C, T, H', W']
            x = x_resized.reshape(B, T, C, target_H, target_W).permute(0, 2, 1, 3, 4)
        
        # InternVideo2 Vision Model expects [B, C, T, H, W]
        # Our x is [B, C, F, H, W], so F is T. Structure matches.
        
        # Ensure device/dtype matches model (do not force GPU here; leave to wrapper for unified management)
        p = next(self.model.parameters())
        x = x.to(device=p.device, dtype=p.dtype)
            
        with torch.no_grad():
            outputs = self.model(x)
            
        features = outputs
        if hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state
        elif isinstance(outputs, (tuple, list)) and len(outputs) > 0 and torch.is_tensor(outputs[0]):
            features = outputs[0]
        
        if features.dim() == 3:
            # Remove CLS token if present (expected_tokens + 1)
            patch_size = 14 
            if hasattr(self.model, 'patch_size'):
                patch_size = self.model.patch_size
            
            num_patches = (self.resolution // patch_size) ** 2
            expected_tokens = T * num_patches
            
            if features.shape[1] == expected_tokens + 1:
                # Remove CLS
                features = features[:, 1:, :]
            elif features.shape[1] == expected_tokens:
                pass
            else:
                # Ambiguous case, maybe F*N is different.
                # Assuming it's already F*N if it doesn't match +1
                pass

        if features.dim() != 3:
            raise ValueError(f"Expected 3D features [B, N, D], got shape: {getattr(features, 'shape', None)}")

        return features

    def get_feature_info(self):
        return {
            'model_name': self.model_name,
            'embed_dim': self.embed_dim,
            'resolution': self.resolution,
        }

if __name__ == "__main__":
    # Test
    try:
        adapter = InternVideoNextAdapter.from_config(
            resolution=224,
            model_name='internvideo2-chat-8b'
        )
        # Dummy input [B, C, F, H, W]
        dummy_input = torch.randn(1, 3, 8, 224, 224)
        print(f"Input shape: {dummy_input.shape}")
        
        output = adapter(dummy_input)
        print(f"Output shape: {output.shape}")
    except Exception as e:
        import traceback
        print("Test failed with exception:")
        traceback.print_exc()
        print(f"Error message: {e}")
