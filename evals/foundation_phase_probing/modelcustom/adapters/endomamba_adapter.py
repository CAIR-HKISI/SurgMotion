"""
EndoMamba Foundation Model Adapter
基于官方EndoMamba仓库定义，支持加载EndoMamba checkpoint
"""

import torch
import torch.nn as nn
from typing import Optional
from pathlib import Path
from functools import partial
from base_adapter import BaseFoundationModelAdapter
from utils import load_and_apply_checkpoint


class EndoMambaAdapter(BaseFoundationModelAdapter):
    """EndoMamba模型的Adapter - 输入格式: [B, C, F, H, W]"""
    
    def __init__(self, model, embed_dim: int, model_name: str):
        super().__init__(model, embed_dim)
        self.model_name = model_name
    
    @classmethod
    def from_config(cls, resolution: int, checkpoint: Optional[str] = None, model_name: str = 'endomamba_small'):
        """
        从配置创建adapter，使用官方EndoMamba定义
        
        Args:
            resolution: 输入分辨率（通常是224）
            checkpoint: checkpoint路径
            model_name: 模型架构名称 ('endomamba_tiny', 'endomamba_small', 'endomamba_middle')
        """
        import sys
        
        # 添加EndoMamba路径到sys.path
        endomamba_path = Path(__file__).parent.parent.parent.parent.parent / "foundation_models" / "EndoMamba" / "videomamba"
        if str(endomamba_path) not in sys.path:
            sys.path.insert(0, str(endomamba_path))
        # 导入官方的模型定义
        from video_sm.models.endomamba import endomamba_small
        
        print(f"Loading EndoMamba model: {model_name}")
        
        try:
            # 根据model_name选择模型架构
            model_factory = {
                'endomamba_small': endomamba_small
            }
            
            if model_name not in model_factory:
                raise ValueError(f"Unknown model_name: {model_name}. Choose from {list(model_factory.keys())}")
            
            # 创建模型（pretrained=False，因为我们稍后会加载checkpoint）
            model = model_factory[model_name](pretrained=False, 
                                              img_size=resolution,
                                              num_classes=0,  # 不使用分类头
                                              with_head=False)  # 不使用head
            
            # 获取embed_dim
            embed_dim_map = {
                'endomamba_small': 384,
            }
            embed_dim = embed_dim_map.get(model_name, 384)
            patch_size = 16
            
            print(f"EndoMamba model created: embed_dim={embed_dim}, patch_size={patch_size}")
            
            # 使用通用工具函数加载checkpoint
            success, info = load_and_apply_checkpoint(
                model=model,
                checkpoint_path=checkpoint,
                default_path="/home/chen_chuxi/NSJepa/ckpts_foundation/endomamba_checkpoint-best.pth",
                strict=False,
                key_prefix_to_remove=None,  # EndoMamba的checkpoint可能不需要移除前缀
                verbose=True
            )
            
            if not success:
                print(f"Warning: {info}")

        except Exception as e:
            print(f"Error loading EndoMamba model: {e}")
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

        # EndoMamba期望输入格式为 [B, C, T, H, W]，其中T是时间维度
        # 我们的输入是 [B, C, F, H, W]，所以F对应T
        # 直接使用，因为维度顺序相同
        
        # 如果尺寸不匹配，进行resize（EndoMamba通常使用224×224）
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
        
        # EndoMamba的forward_features期望输入 [B, C, T, H, W]
        # 我们的输入已经是 [B, C, F, H, W]，F对应T
        # 调用forward_features，注意EndoMamba的forward_features返回 (hidden_states, inference_params)
        with torch.no_grad():
            # forward_features返回 (B, T*N, C) 格式的特征
            features, _ = self.model.forward_features(x, inference_params=None)
        
        # 验证输出形状
        if features.dim() != 2:
            raise ValueError(f"Expected 2D features [B, T*N], got shape: {features.shape}")
        
        # EndoMamba的forward_features返回 (B, T*N, C)，需要转换为 [B, F*N, D]
        # 其中 T 对应我们的 F
        B_out, TN, D = features.shape
        assert B_out == B, f"Batch size mismatch: {B_out} != {B}"
        assert D == self.embed_dim, f"Embed dim mismatch: {D} != {self.embed_dim}"
        
        # features已经是 [B, F*N, D] 格式，直接返回
        return features

    def get_feature_info(self):
        """返回特征提取的信息（用于调试）"""
        return {
            'model_name': self.model_name,
            'embed_dim': self.embed_dim,
            'patch_size': getattr(self.model, 'patch_size', 16),
        }


# 测试代码（可选）
if __name__ == "__main__":
    """
    Input shape: torch.Size([2, 3, 4, 224, 224])
    Output shape: torch.Size([2, 784, 384])  # 对于small模型
    Expected: [2, 4*196, 384] = [2, 784, 384]
    """
    # 测试输入输出格式
    adapter = EndoMambaAdapter.from_config(
        resolution=224,
        checkpoint="/home/chen_chuxi/NSJepa/ckpts_foundation/endomamba_checkpoint-best.pth",
        model_name='endomamba_small'
    )
    
    # 模拟输入: [B=2, C=3, F=4, H=224, W=224]
    dummy_input = torch.randn(2, 3, 4, 224, 224)
    
    print(f"Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = adapter(dummy_input)
    
    print(f"Output shape: {output.shape}")  # 应该是 [2, 4*N, D]
    num_patches_per_frame = (224 // 16) ** 2  # 196 patches per frame
    print(f"Expected: [2, {4 * num_patches_per_frame}, {adapter.embed_dim}]")
    print(f"Feature info: {adapter.get_feature_info()}")