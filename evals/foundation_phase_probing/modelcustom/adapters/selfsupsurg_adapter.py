"""
SelfSupSurg Foundation Model Adapter
Based on official SelfSupSurg definitions, supports loading SelfSupSurg checkpoint
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
from evals.foundation_phase_probing.modelcustom.adapters.utils import load_and_apply_checkpoint


class SelfSupSurgAdapter(BaseFoundationModelAdapter):
    """
    Adapter for SelfSupSurg (ResNet50) image backbone on video input.

    Input format: [B, C, F, H, W]. Feature transform: [B, C, F, H, W] → [B, F, C, H, W] → [B*F, C, H, W],
    then backbone returns [B*F, C', H', W']; flattened to [B, F*N, D] (N=49 for 224×224 layer4).
    """
    
    def __init__(self, model, embed_dim: int, model_name: str):
        super().__init__(model, embed_dim)
        self.model_name = model_name
        
        # ResNet50 layer4 output dimension is 2048
        # We can choose which layer to extract features from
        self.layer_dims = 2048
    
    @classmethod
    def from_config(cls, resolution: int, checkpoint: Optional[str] = None, 
                    model_name: str = 'resnet50'):
        """
        Create adapter from config, using ResNet50 architecture
        
        Args:
            resolution: Input resolution (typically 224)
            checkpoint: Path to checkpoint, default is SelfSupSurg
            model_name: Model architecture name
        """
        import torchvision.models as models
        
        logger.info("Loading SelfSupSurg model: %s (ResNet50)", model_name)
        
        try:
            # Create ResNet50 model
            model = models.resnet50(pretrained=False)
            
            # Remove final fully-connected layer and global average pooling layer
            # We only need the feature extraction part
            model = nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4,
            )
            
            # Determine output dimension based on selected layer
            layer_dims = 2048
            
            embed_dim = 2048
            
            logger.info("ResNet50 created: embed_dim=%s resolution=%s", embed_dim, resolution)
            
            # Use generic utility function to load checkpoint
            success, info = load_and_apply_checkpoint(
                model=model,
                checkpoint_path=checkpoint,
                default_path="ckpts/ckpts_foundation/model_final_checkpoint_dino_surg.torch",
                strict=False,
                key_prefix_to_remove="module.",  # May need to remove 'module.' prefix
                verbose=True
            )
            
            if not success:
                logger.warning("Checkpoint load: %s", info)
            
            # Freeze model parameters
            for param in model.parameters():
                param.requires_grad = False
            model.eval()

        except Exception as e:
            logger.exception("Error loading SelfSupSurg model: %s", e)
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
               - F*N: concatenation of all spatial positions from all frames
               - D: embed_dim (2048 for layer4)
        """
        B, C, F, H, W = x.shape
        
        # Force 224x224 as standard input size
        target_H, target_W = 224, 224
        
        # Resize if dimensions do not match
        if H != target_H or W != target_W:
            import torch.nn.functional as fn
            # [B, C, F, H, W] -> [B*F, C, H, W] -> resize -> [B*F, C, 224, 224]
            x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
            x_resized = fn.interpolate(x_reshaped, size=(target_H, target_W), 
                                       mode='bilinear', align_corners=False)
            # [B*F, C, 224, 224] -> [B, F, C, 224, 224] -> [B, C, F, 224, 224]
            x = x_resized.reshape(B, F, C, target_H, target_W).permute(0, 2, 1, 3, 4)
            H, W = target_H, target_W
        
        # ResNet50 is an image model, need to unfold temporal dimension
        # [B, C, F, H, W] -> [B, F, C, H, W] -> [B*F, C, H, W]
        x = x.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
        x = x.reshape(B * F, C, H, W)  # [B*F, C, H, W]
        
        # Extract features via ResNet50
        with torch.no_grad():
            # ResNet50 forward returns [B*F, C', H', W']
            # e.g. for 224x224 input, layer4 output is [B*F, 2048, 7, 7]
            feature_map = self.model(x)
        
        # feature_map shape: [B*F, C', H', W']
        BF, C_out, H_feat, W_feat = feature_map.shape
        
        # Convert feature map to patch-tokens-like format
        # [B*F, C', H', W'] -> [B*F, C', H'*W'] -> [B*F, H'*W', C']
        features = feature_map.flatten(2)  # [B*F, C', H'*W']
        features = features.permute(0, 2, 1)  # [B*F, H'*W', C']
        
        # Verify output shape
        if features.dim() != 3:
            raise ValueError(f"Expected 3D features [B*F, N, D], got shape: {features.shape}")
        
        # Reshape back to batch and temporal dimensions
        N = H_feat * W_feat  # Number of spatial positions (e.g. 7*7=49)
        D = C_out  # Feature dimension
        
        assert features.shape == (B * F, N, D), f"Shape mismatch: {features.shape} != ({B*F}, {N}, {D})"
        
        # Reshape to [B, F, N, D] then flatten to [B, F*N, D]
        features = features.reshape(B, F, N, D)  # [B, F, N, D]
        features = features.reshape(B, F * N, D)  # [B, F*N, D]
        return features

    def get_feature_info(self):
        """Return feature extraction info (for debugging)"""
        return {
            'model_name': self.model_name,
            'embed_dim': self.embed_dim,
            'architecture': 'ResNet50',
            'pretrain': 'SelfSupSurg',
        }


# Test code (optional)
if __name__ == "__main__":
    """
    Test SelfSupSurg adapter input/output format
    
    Expected:
        Input: [B=2, C=3, F=4, H=224, W=224]
        Output: [2, 4*49, 2048] = [2, 196, 2048]
        (assuming layer4 output is 7x7=49 spatial positions)
    """
    # Test input/output format
    adapter = SelfSupSurgAdapter.from_config(
        resolution=224,
        checkpoint="ckpts/ckpts_foundation/model_final_checkpoint_dino_surg.torch",
        model_name='resnet50'
    )
    
    # Simulate input: [B=2, C=3, F=4, H=224, W=224]
    dummy_input = torch.randn(2, 3, 4, 224, 224)
    
    print(f"\nInput shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = adapter(dummy_input)
    
    print(f"Output shape: {output.shape}")  # Should be [2, 4*49, 2048]
    print(f"Expected: [2, {4 * 7 * 7}, {adapter.embed_dim}]")
    print(f"\nFeature info: {adapter.get_feature_info()}")