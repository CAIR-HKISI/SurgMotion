"""
DINOv2 Foundation Model Adapter
修正版：确保输入输出格式与VisionTransformer一致
"""

import torch
import torch.nn as nn
from typing import Optional
from .base_adapter import BaseFoundationModelAdapter


class DINOv2Adapter(BaseFoundationModelAdapter):
    """DINOv2模型的Adapter - 输入格式: [B, C, F, H, W]"""
    
    def __init__(self, model, embed_dim: int, model_name: str):
        super().__init__(model, embed_dim)
        self.model_name = model_name
    
    @classmethod
    def from_config(cls, resolution: int, checkpoint: Optional[str] = None, model_name: str = 'dinov2_vitl14'):
        """从配置创建adapter"""
        import torch
                
        print(f"Loading DINOv2 model: {model_name}")
        
        try:
            # 从torch.hub加载DINOv2
            model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
            
            # 获取embed_dim和num_heads
            embed_dim = model.embed_dim
            
            print(f"✓ DINOv2 loaded: embed_dim={embed_dim}")
            
            # 如果提供了自定义checkpoint，加载它
            if checkpoint:
                ckpt = torch.load(checkpoint, map_location='cpu')
                model.load_state_dict(ckpt, strict=False)
                print(f"✓ Loaded custom checkpoint from {checkpoint}")

        except Exception as e:
            print(f"Error loading DINOv2 model: {e}")
            raise e
        
        return cls(model, embed_dim, model_name)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, F, H, W] 视频输入（与VisionTransformer格式一致）
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

        # ← 新增：检查并调整分辨率以适配patch_size
        patch_size = getattr(self.model, 'patch_size', 14)
        
        # 计算需要的尺寸（向上取整到patch_size的整数倍）
        #target_H = ((H + patch_size - 1) // patch_size) * patch_size
        #target_W = ((W + patch_size - 1) // patch_size) * patch_size

        target_H = 224
        target_W = 224
        
        # 如果尺寸不匹配，进行resize
        if H != target_H or W != target_W:
            import torch.nn.functional as Fn
            # [B, C, F, H, W] → [B*F, C, H, W] → resize → [B*F, C, H', W']
            x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
            x_resized = Fn.interpolate(x_reshaped, size=(target_H, target_W), 
                                    mode='bilinear', align_corners=False)
            # [B*F, C, H', W'] → [B, F, C, H', W'] → [B, C, F, H', W']
            x = x_resized.reshape(B, F, C, target_H, target_W).permute(0, 2, 1, 3, 4)
            H, W = target_H, target_W
        
        
        # DINOv2是图像模型，需要将时间维度展开
        # [B, C, F, H, W] → [B, F, C, H, W] → [B*F, C, H, W]
        x = x.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
        x = x.reshape(B * F, C, H, W)  # [B*F, C, H, W]
        
        # 通过DINOv2提取特征
        with torch.no_grad():
            # DINOv2 forward 返回不同格式取决于版本
            # 尝试多种方式获取patch tokens
            
            # 方法1: 使用 forward_features (推荐)
            if hasattr(self.model, 'forward_features'):
                features = self.model.forward_features(x)  # 可能返回dict或tensor
            else:
                features = self.model(x)
            
            # 处理不同的输出格式
            if isinstance(features, dict):
                # 情况1: 返回dict，包含不同的keys
                if 'x_norm_patchtokens' in features:
                    features = features['x_norm_patchtokens']  # [B*F, N, D]
                elif 'x_prenorm' in features:
                    # 包含CLS token，需要去掉
                    features = features['x_prenorm'][:, 1:, :]  # [B*F, N, D]
                elif 'x_norm_clstoken' in features or 'x_norm_patchtokens' in features:
                    features = features['x_norm_patchtokens']
                else:
                    # 取第一个tensor value
                    features = list(features.values())[0]
                    # 如果包含CLS token（第一个token），去掉它
                    if features.dim() == 3:
                        features = features[:, 1:, :]  # [B*F, N, D]
            elif isinstance(features, torch.Tensor):
                # 情况2: 直接返回tensor
                if features.dim() == 3:
                    # [B*F, N+1, D] 或 [B*F, N, D]
                    # 检查是否包含CLS token（通常第一个位置）
                    # 安全起见，假设有CLS token并去掉
                    # DINOv2通常不返回CLS token在patch tokens中，所以这里保留所有tokens
                    features = features  # [B*F, N, D]
                elif features.dim() == 2:
                    # [B*F, D] - 只有CLS token，这不是我们要的
                    raise ValueError(f"DINOv2 returned only CLS tokens, expected patch tokens. Shape: {features.shape}")
            else:
                raise ValueError(f"Unexpected DINOv2 output type: {type(features)}")
        
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
            'patch_size': getattr(self.model, 'patch_size', 14),
        }

# 测试代码（可选）
if __name__ == "__main__":
    # 测试输入输出格式
    adapter = DINOv2Adapter.from_config(
        resolution=224,
        checkpoint=None,
        model_name='dinov2_vitl14'
    )
    
    # 模拟输入: [B=2, C=3, F=4, H=224, W=224]
    dummy_input = torch.randn(2, 3, 4, 224, 224)
    
    print(f"Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = adapter(dummy_input)
    
    print(f"Output shape: {output.shape}")  # 应该是 [2, 4*N, D]
    print(f"Expected: [2, {4 * (224//14)**2}, {adapter.embed_dim}]")
    print(f"Feature info: {adapter.get_feature_info()}")