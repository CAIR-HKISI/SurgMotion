"""
EndoViT Foundation Model Adapter
基于官方EndoViT仓库定义，支持加载EndoViT_SPR checkpoint
"""
import sys
sys.path.append(".")

import torch
import torch.nn as nn
from typing import Optional
from pathlib import Path
from functools import partial
from evals.foundation_phase_probing.modelcustom.adapters.base_adapter import BaseFoundationModelAdapter
from evals.foundation_phase_probing.modelcustom.adapters.utils import load_and_apply_checkpoint


class EndoViTAdapter(BaseFoundationModelAdapter):
    """EndoViT模型的Adapter - 输入格式: [B, C, F, H, W]"""
    
    def __init__(self, model, embed_dim: int, model_name: str):
        super().__init__(model, embed_dim)
        self.model_name = model_name
    
    @classmethod
    def from_config(cls, resolution: int, checkpoint: Optional[str] = None, model_name: str = 'vit_base_patch16'):
        """
        从配置创建adapter，使用官方EndoViT定义
        
        Args:
            resolution: 输入分辨率（通常是224）
            checkpoint: checkpoint路径，默认为EndoViT_SPR
            model_name: 模型架构名称 ('vit_base_patch16', 'vit_large_patch16', 'vit_huge_patch14')
        """
        import sys
        
        # 添加EndoViT路径到sys.path
        endovit_path = Path(__file__).parent.parent.parent.parent.parent / "foundation_models" / "EndoViT" / "pretraining" / "mae"
        if str(endovit_path) not in sys.path:
            sys.path.insert(0, str(endovit_path))
        
        # 导入官方的模型定义
        from models_vit import vit_base_patch16, vit_large_patch16, vit_huge_patch14
        
        print(f"Loading EndoViT model: {model_name}")
        
        try:
            model = vit_large_patch16(pool_type='no_pooling')
            embed_dim = 1024
            patch_size = 16
            
            print(f"EndoViT model created: embed_dim={embed_dim}, patch_size={patch_size}")
            
            # 使用通用工具函数加载checkpoint
            success, info = load_and_apply_checkpoint(
                model=model,
                checkpoint_path=checkpoint,
                default_path="ckpts/ckpts_foundation/endovit_SPR.pth",
                strict=False,
                key_prefix_to_remove=None,  # 如果需要移除前缀，在这里设置
                verbose=True
            )
            
            if not success:
                print(f"Warning: {info}")

        except Exception as e:
            print(f"Error loading EndoViT model: {e}")
            raise e
        
        return cls(model, embed_dim, model_name)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, F, H, W] 视频输入
               - B: batch size
               - C: channels (通常是3)
               - F: frames (时间维度)
               - H, W: 高度和宽度
        
        Returns:
            features: [B, F*N, D] 
               - F*N: 所有帧的所有patch tokens拼接
               - D: embed_dim
        """
        B, C, F, H, W = x.shape

        # 强制使用224×224作为EndoViT的标准输入尺寸
        target_H, target_W = 224, 224
        
        # 如果尺寸不匹配，进行resize
        if H != target_H or W != target_W:
            import torch.nn.functional as fn
            #print(f"Resizing input from {H}×{W} to {target_H}×{target_W} (EndoViT standard)")
            # [B, C, F, H, W] → [B*F, C, H, W] → resize → [B*F, C, 224, 224]
            x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
            x_resized = fn.interpolate(x_reshaped, size=(target_H, target_W), 
                                       mode='bilinear', align_corners=False)
            # [B*F, C, 224, 224] → [B, F, C, 224, 224] → [B, C, F, 224, 224]
            x = x_resized.reshape(B, F, C, target_H, target_W).permute(0, 2, 1, 3, 4)
            H, W = target_H, target_W
        
        # EndoViT是图像模型，需要将时间维度展开
        # [B, C, F, H, W] → [B, F, C, H, W] → [B*F, C, H, W]
        x = x.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
        x = x.reshape(B * F, C, H, W)  # [B*F, C, H, W]
        
        # 通过EndoViT提取特征
        with torch.no_grad():
            # 使用官方的forward_features方法
            # pool_type='no_pooling' 返回 [B*F, N, D] (不包括CLS token)
            features = self.model.forward_features(x)
        
        # 验证输出形状
        if features.dim() != 3:
            raise ValueError(f"Expected 3D features [B*F, N, D], got shape: {features.shape}")
        
        # 重组回batch和时间维度
        BF, N, D = features.shape
        assert BF == B * F, f"Shape mismatch: {BF} != {B} * {F}"
        
        # 重塑为 [B, F, N, D] 然后展平为 [B, F*N, D]
        features = features.reshape(B, F, N, D)  # [B, F, N, D]
        features = features.reshape(B, F * N, D)  # [B, F*N, D]
        
        return features

    def get_feature_info(self):
        """返回特征提取的信息（用于调试）"""
        return {
            'model_name': self.model_name,
            'embed_dim': self.embed_dim,
            'patch_size': getattr(self.model, 'patch_size', 16),
            'pool_type': getattr(self.model, 'pool_type', 'no_pooling'),
        }


# 测试代码（可选）
if __name__ == "__main__":
    """
    Input shape: torch.Size([2, 3, 4, 224, 224])
    Output shape: torch.Size([2, 784, 1024])
    Expected: [2, 784, 1024]
    """
    # 测试输入输出格式
    adapter = EndoViTAdapter.from_config(
        resolution=224,
        checkpoint="ckpts/ckpts_foundation/endovit_SPR.pth",
        model_name='vit_base_patch16'
    )
    
    # 模拟输入: [B=2, C=3, F=4, H=224, W=224]
    dummy_input = torch.randn(2, 3, 4, 224, 224)
    
    print(f"Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = adapter(dummy_input)
    
    print(f"Output shape: {output.shape}")  # 应该是 [2, 4*N, D]
    print(f"Expected: [2, {4 * (224//16)**2}, {adapter.embed_dim}]")
    print(f"Feature info: {adapter.get_feature_info()}")