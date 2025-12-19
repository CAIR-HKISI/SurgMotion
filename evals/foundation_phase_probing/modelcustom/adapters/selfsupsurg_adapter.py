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

class SelfSupSurgAdapter(BaseFoundationModelAdapter):
    """SelfSupSurg模型的Adapter - 输入格式: [B, C, F, H, W]"""
    
    def __init__(self, model, embed_dim: int, model_name: str):
        super().__init__(model, embed_dim)
        self.model_name = model_name
        
        # ResNet50的layer4输出维度是2048
        # 我们可以选择在哪一层提取特征
        self.layer_dims = 2048
    
    @classmethod
    def from_config(cls, resolution: int, checkpoint: Optional[str] = None, 
                    model_name: str = 'resnet50'):
        """
        从配置创建adapter，使用ResNet50架构
        
        Args:
            resolution: 输入分辨率（通常是224）
            checkpoint: checkpoint路径，默认为SelfSupSurg
            model_name: 模型架构名称
        """
        import torchvision.models as models
        
        print(f"Loading SelfSupSurg model: {model_name} (ResNet50)")
        
        try:
            # 创建 ResNet50 模型
            model = models.resnet50(pretrained=False)
            
            # 移除最后的全连接层和全局平均池化层
            # 我们只需要特征提取部分
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
            
            # 根据选择的层确定输出维度
            layer_dims = 2048
            
            embed_dim = 2048
            
            print(f"✓ ResNet50 architecture created:")
            print(f"  - embed_dim: {embed_dim}")
            print(f"  - resolution: {resolution}")
            
            # 使用通用工具函数加载checkpoint
            success, info = load_and_apply_checkpoint(
                model=model,
                checkpoint_path=checkpoint,
                default_path="ckpts/ckpts_foundation/model_final_checkpoint_dino_surg.torch",
                strict=False,
                key_prefix_to_remove="module.",  # 可能需要移除 'module.' 前缀
                verbose=True
            )
            
            if not success:
                print(f"Warning: {info}")
            
            # 冻结模型参数
            for param in model.parameters():
                param.requires_grad = False
            model.eval()

        except Exception as e:
            print(f"Error loading SelfSupSurg model: {e}")
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
               - F*N: 所有帧的所有spatial位置拼接
               - D: embed_dim (2048 for layer4)
        """
        B, C, F, H, W = x.shape
        
        # 强制使用224×224作为标准输入尺寸
        target_H, target_W = 224, 224
        
        # 如果尺寸不匹配，进行resize
        if H != target_H or W != target_W:
            import torch.nn.functional as fn
            print(f"Resizing input from {H}×{W} to {target_H}×{target_W} (ResNet50 standard)")
            # [B, C, F, H, W] → [B*F, C, H, W] → resize → [B*F, C, 224, 224]
            x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
            x_resized = fn.interpolate(x_reshaped, size=(target_H, target_W), 
                                       mode='bilinear', align_corners=False)
            # [B*F, C, 224, 224] → [B, F, C, 224, 224] → [B, C, F, 224, 224]
            x = x_resized.reshape(B, F, C, target_H, target_W).permute(0, 2, 1, 3, 4)
            H, W = target_H, target_W
        
        # ResNet50是图像模型，需要将时间维度展开
        # [B, C, F, H, W] → [B, F, C, H, W] → [B*F, C, H, W]
        x = x.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
        x = x.reshape(B * F, C, H, W)  # [B*F, C, H, W]
        
        # 通过ResNet50提取特征
        with torch.no_grad():
            # ResNet50 forward 返回 [B*F, C', H', W']
            # 例如对于224x224输入，layer4输出是 [B*F, 2048, 7, 7]
            feature_map = self.model(x)
        
        # feature_map shape: [B*F, C', H', W']
        BF, C_out, H_feat, W_feat = feature_map.shape
        
        # 将特征图转换为类似patch tokens的格式
        # [B*F, C', H', W'] → [B*F, C', H'*W'] → [B*F, H'*W', C']
        features = feature_map.flatten(2)  # [B*F, C', H'*W']
        features = features.permute(0, 2, 1)  # [B*F, H'*W', C']
        
        # 验证输出形状
        if features.dim() != 3:
            raise ValueError(f"Expected 3D features [B*F, N, D], got shape: {features.shape}")
        
        # 重组回batch和时间维度
        N = H_feat * W_feat  # 空间位置数 (例如 7*7=49)
        D = C_out  # 特征维度
        
        assert features.shape == (B * F, N, D), f"Shape mismatch: {features.shape} != ({B*F}, {N}, {D})"
        
        # 重塑为 [B, F, N, D] 然后展平为 [B, F*N, D]
        features = features.reshape(B, F, N, D)  # [B, F, N, D]
        features = features.reshape(B, F * N, D)  # [B, F*N, D]
        
        print(f"📊 SelfSupSurg feature extraction:")
        print(f"   Input: [{B}, {C}, {F}, {H}, {W}]")
        print(f"   Feature map: [{BF}, {C_out}, {H_feat}, {W_feat}]")
        print(f"   Output: [{B}, {F*N}, {D}] (F={F}, N={N}, D={D})")
        
        return features

    def get_feature_info(self):
        """返回特征提取的信息（用于调试）"""
        return {
            'model_name': self.model_name,
            'embed_dim': self.embed_dim,
            'architecture': 'ResNet50',
            'pretrain': 'SelfSupSurg',
        }


# 测试代码（可选）
if __name__ == "__main__":
    """
    测试SelfSupSurg adapter的输入输出格式
    
    Expected:
        Input: [B=2, C=3, F=4, H=224, W=224]
        Output: [2, 4*49, 2048] = [2, 196, 2048]
        (假设layer4输出是7x7=49个spatial位置)
    """
    # 测试输入输出格式
    adapter = SelfSupSurgAdapter.from_config(
        resolution=224,
        checkpoint="ckpts/ckpts_foundation/model_final_checkpoint_dino_surg.torch",
        model_name='resnet50'
    )
    
    # 模拟输入: [B=2, C=3, F=4, H=224, W=224]
    dummy_input = torch.randn(2, 3, 4, 224, 224)
    
    print(f"\nInput shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = adapter(dummy_input)
    
    print(f"Output shape: {output.shape}")  # 应该是 [2, 4*49, 2048]
    print(f"Expected: [2, {4 * 7 * 7}, {adapter.embed_dim}]")
    print(f"\nFeature info: {adapter.get_feature_info()}")