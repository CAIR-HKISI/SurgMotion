"""
GastroNet Foundation Model Adapter
基于DINOv1 ViT-small架构，使用GastroNet-5M预训练权重
"""
import sys
sys.path.append(".")

import torch
import torch.nn as nn
from typing import Optional
from pathlib import Path
from evals.foundation_phase_probing.modelcustom.adapters.base_adapter import BaseFoundationModelAdapter

class GastroNetAdapter(BaseFoundationModelAdapter):
    """GastroNet模型的Adapter (DINOv1 ViT-small) - 输入格式: [B, C, F, H, W]"""
    
    def __init__(self, model, embed_dim: int, model_name: str):
        super().__init__(model, embed_dim)
        self.model_name = model_name
        self.patch_size = 16  # ViT-small standard
    
    @classmethod
    def from_config(cls, resolution: int, checkpoint: Optional[str] = None, model_name: str = 'vit_small_patch16'):
        """
        从配置创建adapter，使用DINOv1 ViT-small架构
        
        Args:
            resolution: 输入分辨率（通常是224）
            checkpoint: checkpoint路径，默认为GastroNet-5M
            model_name: 模型架构名称
        """
        print(f"Loading GastroNet model: {model_name} (DINOv1 ViT-small)")
            
        try:
            # 方法2: 使用torch.hub加载DINOv1
            model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False)
            embed_dim = 384
            patch_size = 16
                
            # 加载GastroNet权重
            ckpt = torch.load(
                checkpoint or "ckpts/ckpts_foundation/VITS_GastroNet-5M_DINOv1.pth",
                map_location='cpu',
                weights_only=False  # GastroNet可能包含非tensor数据
            )
                
            # 处理不同的checkpoint格式
            if isinstance(ckpt, dict):
                if 'state_dict' in ckpt:
                    state_dict = ckpt['state_dict']
                elif 'model' in ckpt:
                    state_dict = ckpt['model']
                elif 'teacher' in ckpt:
                    state_dict = ckpt['teacher']
                else:
                    state_dict = ckpt
            else:
                state_dict = ckpt
                
            # 移除可能的前缀
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k
                # 移除常见的前缀
                for prefix in ['module.', 'backbone.', 'encoder.']:
                    if new_k.startswith(prefix):
                        new_k = new_k[len(prefix):]
                new_state_dict[new_k] = v
                
            # 加载权重
            msg = model.load_state_dict(new_state_dict, strict=False)
            print(f"✓ Loaded GastroNet weights")
            if msg.missing_keys:
                print(f"  Missing keys: {len(msg.missing_keys)}")
            if msg.unexpected_keys:
                print(f"  Unexpected keys: {len(msg.unexpected_keys)}")
                    
        except Exception as e2:
            print(f"Alternative method also failed: {e2}")
            raise e2
        
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
               - D: embed_dim (384 for ViT-small)
        """
        B, C, F, H, W = x.shape
    
        target_H, target_W = 224, 224
        
        # 如果尺寸不匹配，进行resize
        if H != target_H or W != target_W:
            import torch.nn.functional as fn
            print(f"Resizing input from {H}×{W} to {target_H}×{target_W} (DINOv1 standard)")
            # [B, C, F, H, W] → [B*F, C, H, W] → resize → [B*F, C, 224, 224]
            x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
            x_resized = fn.interpolate(x_reshaped, size=(target_H, target_W), 
                                    mode='bilinear', align_corners=False)
            # [B*F, C, 224, 224] → [B, F, C, 224, 224] → [B, C, F, 224, 224]
            x = x_resized.reshape(B, F, C, target_H, target_W).permute(0, 2, 1, 3, 4)
            H, W = target_H, target_W
        
        # DINOv1是图像模型，需要将时间维度展开
        # [B, C, F, H, W] → [B, F, C, H, W] → [B*F, C, H, W]
        x = x.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
        x = x.reshape(B * F, C, H, W)  # [B*F, C, H, W]
        
        # 通过DINOv1 GastroNet提取特征
        with torch.no_grad():
            # 根据DINOv1官方代码，使用get_intermediate_layers获取最后一层的所有tokens
            # 这会返回 [B*F, N+1, D]，其中包含CLS token和所有patch tokens
            # get_intermediate_layers(x, n=1) 返回最后1层的输出
            # 返回格式: list of [B*F, N+1, D]
            output = self.model.get_intermediate_layers(x, n=1)
            features = output[0]  # 取第一个（也是唯一一个）: [B*F, N+1, D]
            # 移除CLS token（第一个位置）
            features = features[:, 1:, :]  # [B*F, N, D]
        
        # 验证输出形状
        if features.dim() != 3:
            raise ValueError(f"Expected 3D features [B*F, N, D], got shape: {features.shape}")
        
        BF, N, D = features.shape
        
        # 验证维度
        assert BF == B * F, f"Batch×Frames mismatch: {BF} != {B} * {F}"
        assert D == self.embed_dim, f"Embed dim mismatch: {D} != {self.embed_dim}"
        
        expected_num_patches = (target_H // self.patch_size) * (target_W // self.patch_size)
        assert N == expected_num_patches, (
            f"Patch number mismatch: {N} != {expected_num_patches} "
            f"(expected {target_H}/{self.patch_size} × {target_W}/{self.patch_size})"
        )
        
        # 重塑为 [B, F, N, D] 然后展平为 [B, F*N, D]
        features = features.reshape(B, F, N, D)  # [B, F, N, D]
        features = features.reshape(B, F * N, D)  # [B, F*N, D]
        
        return features

    def get_feature_info(self):
        """返回特征提取的信息（用于调试）"""
        return {
            'model_name': self.model_name,
            'embed_dim': self.embed_dim,
            'patch_size': self.patch_size,
            'architecture': 'DINOv1 ViT-small',
            'pretrain': 'GastroNet-5M',
        }


# 测试代码（可选）
if __name__ == "__main__":
    """
    测试GastroNet adapter的输入输出格式
    
    Expected:
        Input: [B=2, C=3, F=4, H=224, W=224]
        Output: [2, 4*196, 384] = [2, 784, 384]
    """
    # 测试输入输出格式
    adapter = GastroNetAdapter.from_config(
        resolution=224,
        checkpoint="ckpts/ckpts_foundation/VITS_GastroNet-5M_DINOv1.pth",
        model_name='vit_small_patch16'
    )
    
    # 模拟输入: [B=2, C=3, F=4, H=224, W=224]
    dummy_input = torch.randn(2, 3, 4, 224, 224)
    
    print(f"\nInput shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = adapter(dummy_input)
    
    print(f"Output shape: {output.shape}")  # 应该是 [2, 4*196, 384]
    print(f"Expected: [2, {4 * (224//16)**2}, {adapter.embed_dim}]")
    print(f"\nFeature info: {adapter.get_feature_info()}")