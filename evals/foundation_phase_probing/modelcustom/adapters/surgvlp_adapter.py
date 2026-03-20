import os
import sys

# Add foundation_models/SurgVLP to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming repository root is 4 levels up from this file
# evals/foundation_phase_probing/modelcustom/adapters/surgvlp_adapter.py -> root
repo_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
surgvlp_path = os.path.join(repo_root, "foundation_models/SurgVLP")

if surgvlp_path not in sys.path:
    # Insert at the front to avoid being overridden by a same-named "surgvlp" package in the environment
    sys.path.insert(0, surgvlp_path)

import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from typing import Optional
import importlib
import importlib.util

try:
    import transformers
except ImportError:
    from unittest.mock import MagicMock
    m = MagicMock()
    m.__file__ = "dummy_transformers"
    sys.modules["transformers"] = m
    sys.modules["transformers.models"] = m
    sys.modules["transformers.models.bert"] = m
    logger.warning("Injected Mock transformers into sys.modules to work around environment error.")
except Exception:
    from unittest.mock import MagicMock
    m = MagicMock()
    sys.modules["transformers"] = m
    logger.warning("Forcibly injected Mock transformers.")

if "surgvlp" in sys.modules:
    _loaded = sys.modules["surgvlp"]
    if not hasattr(_loaded, "load"):
        logger.warning("Detected incompatible surgvlp in sys.modules (%s), removing to reload", getattr(_loaded, '__file__', 'unknown'))
        del sys.modules["surgvlp"]
        for k in list(sys.modules.keys()):
            if k.startswith("surgvlp."):
                del sys.modules[k]

try:
    import surgvlp
    if not hasattr(surgvlp, "load"):
        import surgvlp.surgvlp
        if hasattr(surgvlp.surgvlp, "load"):
             surgvlp = surgvlp.surgvlp
    
    _SURGVLP_IMPORT_ERROR: Optional[BaseException] = None
except Exception as e:
    surgvlp = None  # type: ignore[assignment]
    _SURGVLP_IMPORT_ERROR = e
    logger.warning("Could not import surgvlp from %s: %s: %s", surgvlp_path, type(e).__name__, e)

from evals.foundation_phase_probing.modelcustom.adapters.base_adapter import BaseFoundationModelAdapter

class SurgVLPAdapter(BaseFoundationModelAdapter):
    """
    Adapter for SurgVLP (ResNet50 image backbone) on video input.

    Input format: [B, C, F, H, W]. Feature transform: [B, C, F, H, W] → [B, F, C, H, W] → [B*F, C, H, W],
    then backbone returns spatial feature map, flattened to [B, F*N, D].
    """
    
    def __init__(self, model, embed_dim: int, model_name: str, resolution: int = 224):
        super().__init__(model, embed_dim)
        self.model_name = model_name
        self.resolution = resolution
        
    @classmethod
    def from_config(cls, resolution: int = 224, checkpoint: Optional[str] = None, model_name: str = 'SurgVLP'):
        """Create adapter from config"""
        
        if surgvlp is None:
            missing_files = []
            target_pkg_dir = os.path.join(surgvlp_path, "surgvlp")
            
            if not os.path.exists(os.path.join(target_pkg_dir, "codes")):
                missing_files.append("surgvlp/codes/ (directory missing)")
            if not os.path.exists(os.path.join(target_pkg_dir, "__init__.py")):
                missing_files.append("surgvlp/__init__.py (file missing)")
            
            hint = ""
            if missing_files:
                hint = f"\n** Detected missing files: {', '.join(missing_files)} **\nThis is usually caused by incomplete file sync/upload."

            logger.warning(
                "Could not import surgvlp, will try to continue (may cause subsequent crash). Original: %s: %s%s",
                type(_SURGVLP_IMPORT_ERROR).__name__, _SURGVLP_IMPORT_ERROR, hint
            )
        
        model_config = dict(
            type='SurgVLP',
            backbone_img = dict(
                type='img_backbones/ImageEncoder',
                num_classes=768,
                pretrained='imagenet',
                backbone_name='resnet_50',
                img_norm=False
            ),
            # backbone_text removed to avoid dependency on transformers
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if checkpoint is None:
            checkpoint = "ckpts/ckpts_foundation/PeskaVLP.pth"
        
        logger.info("Loading SurgVLP model from %s", checkpoint)
        # Load model using surgvlp.load, pretrain arg can be used to load specific checkpoint
        model, _ = surgvlp.load(model_config, device=device, pretrain=checkpoint)
        
        # Determine embed_dim, ResNet50 layer4 output channels is 2048
        embed_dim = 2048
        
        logger.info("SurgVLP loaded: embed_dim=%s", embed_dim)
        
        return cls(model, embed_dim, model_name, resolution)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, F, H, W] video input
               - B: batch size, C: channels (3), F: frames, H, W: height and width

        Returns:
            features: [B, F*N, D]
               - F*N: flattened spatial positions from all frames; D: embed_dim (2048)
        """
        B, C, F, H, W = x.shape
        
        target_H = self.resolution
        target_W = self.resolution
        
        # [B, C, F, H, W] → [B, F, C, H, W] → [B*F, C, H, W]
        x = x.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
        
        # Resize if needed
        if H != target_H or W != target_W:
            import torch.nn.functional as Fn
            x = Fn.interpolate(
                x, 
                size=(target_H, target_W), 
                mode='bilinear', 
                align_corners=False
            )
            
        # Extract features using internal ResNet backbone
        # model -> backbone_img (ImageEncoder) -> model (ResNet)
        if hasattr(self.model, 'backbone_img') and hasattr(self.model.backbone_img, 'model'):
            resnet = self.model.backbone_img.model
            
            # Forward pass through ResNet up to layer4
            x = resnet.conv1(x)
            x = resnet.bn1(x)
            x = resnet.relu(x)
            x = resnet.maxpool(x)

            x = resnet.layer1(x)
            x = resnet.layer2(x)
            x = resnet.layer3(x)
            x = resnet.layer4(x)
            
            # Output shape: [B*F, 2048, H/32, W/32]
            
            # Flatten spatial dimensions: [B*F, C, H', W'] -> [B*F, C, N] -> [B*F, N, C]
            x = x.flatten(2).transpose(1, 2)
            
        else:
            # Fallback if structure is different (should not happen with standard SurgVLP)
            raise AttributeError("Could not find ResNet backbone in SurgVLP model")
            
        # Reshape back to include time dimension
        # [B*F, N, D] -> [B, F, N, D] -> [B, F*N, D]
        BF, N, D = x.shape
        x = x.reshape(B, F, N, D).reshape(B, F * N, D)
        
        return x

    def get_feature_info(self):
        return {
            'model_name': self.model_name,
            'embed_dim': self.embed_dim,
            'resolution': self.resolution
        }

if __name__ == "__main__":
    # Test code
    try:
        adapter = SurgVLPAdapter.from_config(resolution=224)
        dummy_input = torch.randn(2, 3, 4, 224, 224).cuda()
        with torch.no_grad():
            output = adapter(dummy_input)
        print(f"Input: {dummy_input.shape}")
        print(f"Output: {output.shape}")
    except Exception as e:
        print(f"Test failed: {e}")
