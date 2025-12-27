import torch
import torch.nn as nn
import sys
import os
from typing import Optional

# Add foundation_models/SurgVLP to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming repository root is 4 levels up from this file
# evals/foundation_phase_probing/modelcustom/adapters/surgvlp_adapter.py -> root
repo_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
surgvlp_path = os.path.join(repo_root, "foundation_models/SurgVLP")

if surgvlp_path not in sys.path:
    sys.path.append(surgvlp_path)

try:
    import surgvlp
    from mmengine.config import Config
except ImportError:
    print(f"Warning: Could not import surgvlp from {surgvlp_path}")

from .base_adapter import BaseFoundationModelAdapter

class SurgVLPAdapter(BaseFoundationModelAdapter):
    """SurgVLP Adapter - Input format: [B, C, F, H, W]"""
    
    def __init__(self, model, embed_dim: int, model_name: str, resolution: int = 224):
        super().__init__(model, embed_dim)
        self.model_name = model_name
        self.resolution = resolution
        
    @classmethod
    def from_config(cls, resolution: int = 224, checkpoint: Optional[str] = None, model_name: str = 'SurgVLP'):
        """Create adapter from config"""
        
        # Define config directly to avoid mmengine config parsing issues with transforms
        model_config = dict(
            type='SurgVLP',
            backbone_img = dict(
                type='img_backbones/ImageEncoder',
                num_classes=768,
                pretrained='imagenet',
                backbone_name='resnet_50',
                img_norm=False
            ),
            backbone_text= dict(
                type='text_backbones/BertEncoder',
                text_bert_type='emilyalsentzer/Bio_ClinicalBERT',
                text_last_n_layers=4,
                text_aggregate_method='sum',
                text_norm=False,
                text_embedding_dim=768,
                text_freeze_bert=False,
                text_agg_tokens=True
            )
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if checkpoint is None:
            checkpoint = "ckpts/ckpts_foundation/SurgVLP.pth"
        
        print(f"Loading SurgVLP model from {checkpoint}...")
        # Load model using surgvlp.load
        # pretrain arg can be used to load specific checkpoint
        model, _ = surgvlp.load(model_config, device=device, pretrain=checkpoint)
        
        # Determine embed_dim
        # ResNet50 layer4 output channels is 2048
        embed_dim = 2048
        
        print(f"✓ SurgVLP loaded: embed_dim={embed_dim}")
        
        return cls(model, embed_dim, model_name, resolution)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, F, H, W] Video input
        Returns:
            features: [B, F*N, D]
        """
        B, C, F, H, W = x.shape
        
        target_H = self.resolution
        target_W = self.resolution
        
        # Flatten time dimension for processing
        # [B, C, F, H, W] -> [B*F, C, H, W]
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
