"""
SurgeNet Foundation Model Adapter

Based on SurgeNet repository (https://github.com/TimJaspers0801/SurgeNet)
Supports CAFormer-S18 backbone and its variants (ConvNextv2, PVTv2)

SurgeNet is a model pre-trained with DINO self-supervised learning on large-scale surgical video data,
and performs well on tasks such as surgical scene segmentation and phase recognition.
"""

import sys
if "." not in sys.path:
    sys.path.append(".")

import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from typing import Optional
from pathlib import Path

from evals.foundation_phase_probing.modelcustom.adapters.base_adapter import BaseFoundationModelAdapter

# SurgeNet pretrained weight URLs
SURGENET_URLS = {
    "surgenetxl": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/SurgeNetXL_checkpoint_epoch0050_teacher.pth?download=true",
    "surgenet": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/SurgeNet_checkpoint_epoch0050_teacher.pth?download=true",
    "surgenet_small": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/SurgeNetSmall_checkpoint_epoch0050_teacher.pth?download=true",
    "surgenet_cholec": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/CHOLEC_checkpoint_epoch0050_teacher.pth?download=true",
    "surgenet_ramie": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/RAMIE_checkpoint_epoch0050_teacher.pth?download=true",
    "surgenet_rarp": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/RARP_checkpoint_epoch0050_teacher.pth?download=true",
    "surgenet_public": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/Public_checkpoint0050.pth?download=true",
    "surgenet_convnextv2": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/SurgeNet_ConvNextv2_checkpoint_epoch0050_teacher.pth?download=true",
    "surgenet_pvtv2": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/SurgeNet_PVTv2_checkpoint_epoch0050_teacher.pth?download=true",
}

# Local weight filename mapping
SURGENET_FILENAMES = {
    "surgenetxl": "SurgeNetXL_checkpoint_epoch0050_teacher.pth",
    "surgenet_convnextv2": "SurgeNet_ConvNextv2_checkpoint_epoch0050_teacher.pth",
}

# Local weight directory
LOCAL_CHECKPOINT_DIR = Path("ckpts/ckpts_foundation/SurgeNetModels")

# Output dimensions for different backbones
BACKBONE_DIMS = {
    "caformer": 512,       # CAFormer-S18
    "convnextv2": 768,     # ConvNextv2-Tiny
    "pvtv2": 512,          # PVTv2-B2
}


class SurgeNetAdapter(BaseFoundationModelAdapter):
    """
    SurgeNet model Adapter - Input format: [B, C, F, H, W]
    
    SurgeNet uses CAFormer-S18 (MetaFormer architecture) as the default backbone,
    and also supports ConvNextv2 and PVTv2 as alternative backbones.
    
    Model output dimensions:
    - CAFormer-S18: 512
    - ConvNextv2-Tiny: 768
    - PVTv2-B2: 512
    """
    
    def __init__(self, model, embed_dim: int, model_name: str, backbone_type: str = "caformer"):
        super().__init__(model, embed_dim)
        self.model_name = model_name
        self.backbone_type = backbone_type
        self._first_forward = True  # Used to control log output
    
    @classmethod
    def from_config(
        cls,
        resolution: int,
        checkpoint: Optional[str] = None,
        model_name: str = 'surgenetxl',
        backbone_type: Optional[str] = None
    ):
        """
        Create SurgeNet adapter from config
        
        Args:
            resolution: Input resolution (224 recommended)
            checkpoint: Local checkpoint path (if None, download from HuggingFace)
            model_name: Model name
            backbone_type: Backbone type ('caformer', 'convnextv2', 'pvtv2').
                          If None, will try to infer from model_name.
        
        Returns:
            SurgeNetAdapter instance
        """
        # 1. Infer backbone_type
        if backbone_type is None:
            if 'convnextv2' in model_name.lower():
                backbone_type = 'convnextv2'
            elif 'pvtv2' in model_name.lower():
                backbone_type = 'pvtv2'
            else:
                backbone_type = 'caformer'
                
        logger.info("Loading SurgeNet model: %s (backbone: %s)", model_name, backbone_type)
        
        # 2. Try to auto-find local weights
        if checkpoint is None:
            model_key = model_name.lower().replace('-', '_')
            if model_key in SURGENET_FILENAMES:
                local_path = LOCAL_CHECKPOINT_DIR / SURGENET_FILENAMES[model_key]
                if local_path.exists():
                    checkpoint = str(local_path)
                    logger.info("Found local checkpoint: %s", checkpoint)
        
        # 3. Add SurgeNet repository path to sys.path
        surgenet_path = Path(__file__).parent.parent.parent.parent.parent / "foundation_models" / "SurgeNet"
        if str(surgenet_path) not in sys.path:
            sys.path.insert(0, str(surgenet_path))
            logger.debug("Added SurgeNet path: %s", surgenet_path)
        
        try:
            # 4. Load model according to backbone type
            if backbone_type == 'caformer':
                from foundation_models.SurgeNet.metaformer import caformer_s18
                
                # Determine weight URL
                model_key = model_name.lower().replace('-', '_')
                weights_url = SURGENET_URLS.get(model_key, SURGENET_URLS.get('surgenetxl'))
                
                # If local checkpoint exists, do not load via URL
                if checkpoint and Path(checkpoint).exists():
                    weights_url = None
                    logger.info("Using local checkpoint: %s", checkpoint)
                
                # Create model
                model = caformer_s18(
                    num_classes=0,
                    pretrained='SurgeNet',
                    pretrained_weights=weights_url
                )
                
                embed_dim = BACKBONE_DIMS['caformer']
                
            elif backbone_type == 'convnextv2':
                from foundation_models.SurgeNet.convnextv2 import convnextv2_tiny
                
                weights_url = SURGENET_URLS.get('surgenet_convnextv2')
                
                # If local checkpoint exists, do not load via URL
                if checkpoint and Path(checkpoint).exists():
                    weights_url = None
                
                # Create model (pretrained_weights=None to avoid auto-download)
                model = convnextv2_tiny(
                    num_classes=0,
                    pretrained_weights=None if checkpoint else weights_url
                )
                
                embed_dim = BACKBONE_DIMS['convnextv2']
                
            elif backbone_type == 'pvtv2':
                from foundation_models.SurgeNet.pvtv2 import pvt_v2_b2
                
                weights_url = SURGENET_URLS.get('surgenet_pvtv2')
                if checkpoint and Path(checkpoint).exists():
                    weights_url = checkpoint # pvt_v2 implementation handles path string as local path too usually
                
                model = pvt_v2_b2(
                    num_classes=0,
                    pretrained_weights=weights_url
                )
                embed_dim = BACKBONE_DIMS['pvtv2']
            
            else:
                raise ValueError(f"Unsupported backbone type: {backbone_type}")
            
            # 5. Load local weights (generic logic)
            if checkpoint and Path(checkpoint).exists():
                state_dict = torch.load(checkpoint, map_location='cpu')
                
                # Handle possible nested structure
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif 'model' in state_dict:
                    state_dict = state_dict['model']
                
                # Remove classification head weights
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head.')}
                
                # Remove possible prefixes (for CAFormer etc.)
                for prefix in ['module.', 'backbone.', 'encoder.']:
                    state_dict = {k.replace(prefix, ''): v for k, v in state_dict.items()}
                
                msg = model.load_state_dict(state_dict, strict=False)
                logger.info("Loaded checkpoint: %s", msg)
            
            logger.info(
                "SurgeNet loaded: model=%s backbone=%s embed_dim=%s",
                model_name, backbone_type, embed_dim
            )
            
        except Exception as e:
            logger.exception("Error loading SurgeNet model: %s", e)
            import traceback
            traceback.print_exc()
            raise e
        
        return cls(model, embed_dim, model_name, backbone_type)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, F, H, W] video input
               - B: batch size, C: channels (3), F: frames, H, W: height and width

        Returns:
            features: [B, F*N, D]
               - F*N: flattened spatial feature positions from all frames; D: backbone embed_dim
        """
        B, C, F, H, W = x.shape
        
        target_H, target_W = 224, 224
        if H != target_H or W != target_W:
            import torch.nn.functional as fn
            # [B, C, F, H, W] → [B, F, C, H, W] → [B*F, C, H, W] → resize → [B, C, F, H', W']
            x = x.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
            x = fn.interpolate(x, size=(target_H, target_W), mode='bilinear', align_corners=False)
            x = x.reshape(B, F, C, target_H, target_W).permute(0, 2, 1, 3, 4)
            H, W = target_H, target_W
        
        # [B, C, F, H, W] → [B, F, C, H, W] → [B*F, C, H, W]
        x = x.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
        
        # Extract features via SurgeNet
        with torch.no_grad():
            if self.backbone_type == 'convnextv2':
                 features = self.model.forward_features(x)
                 # Handle varied return types (tensor, tuple, list)
                 if isinstance(features, (tuple, list)):
                     feature_map = features[-1] if isinstance(features[-1], list) else features[-1]
                     if isinstance(feature_map, (tuple, list)): # Nested list check
                         feature_map = feature_map[-1]
                 else:
                     feature_map = features
            else:
                # CAFormer returns (global_features, feature_list)
                outputs = self.model.forward_features(x)
                if isinstance(outputs, tuple) and len(outputs) == 2:
                    global_features, feature_list = outputs
                    feature_map = feature_list[-1]
                else:
                    feature_map = outputs

        # feature_map shape: [B*F, C', H', W']
        BF, C_out, H_feat, W_feat = feature_map.shape
        
        # Flatten spatial dimensions: [B*F, C', H', W'] -> [B*F, C', N] -> [B*F, N, C']
        features = feature_map.flatten(2).permute(0, 2, 1)
        
        # Restructure to [B, F*N, D]
        N = H_feat * W_feat
        features = features.reshape(B, F * N, C_out)
        self._first_forward = False
        return features
    
    def get_feature_info(self):
        return {
            'model_name': self.model_name,
            'embed_dim': self.embed_dim,
            'backbone': self.backbone_type,
            'pretrain': 'DINO (SurgeNet)',
        }


if __name__ == "__main__":
    print("Testing SurgeNet Adapter")
    
    # 1. Test SurgeNetXL (CAFormer)
    print("\n1. Testing SurgeNetXL (CAFormer)...")
    try:
        adapter = SurgeNetAdapter.from_config(resolution=224, model_name='surgenetxl')
        dummy_input = torch.randn(1, 3, 4, 224, 224)
        output = adapter(dummy_input)
        print(f"  ✓ Output shape: {output.shape} (Expected: [1, 196, 512])")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()

    # 2. Test SurgeNet ConvNextv2
    print("\n2. Testing SurgeNet ConvNextv2...")
    try:
        adapter = SurgeNetAdapter.from_config(resolution=224, model_name='surgenet_convnextv2')
        dummy_input = torch.randn(1, 3, 4, 224, 224)
        output = adapter(dummy_input)
        print(f"  ✓ Output shape: {output.shape} (Expected: [1, 196, 768])")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
