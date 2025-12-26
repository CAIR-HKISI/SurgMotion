"""
DINOv3 Foundation Model Adapter
修正版：确保输入输出格式与VisionTransformer一致
"""

import torch
import torch.nn as nn
from typing import Optional
from .base_adapter import BaseFoundationModelAdapter


class DINOv3Adapter(BaseFoundationModelAdapter):
    """DINOv3模型的Adapter - 输入格式: [B, C, F, H, W]"""
    
    def __init__(self, model, embed_dim: int, model_name: str):
        super().__init__(model, embed_dim)
        self.model_name = model_name
        self.embed_dim = 1024
    
    @classmethod
    def from_config(cls, resolution: int, checkpoint: Optional[str] = None, model_name: str = 'dinov3_vitl14'):
        """从配置创建adapter"""
        import torch
        from modelscope import AutoImageProcessor, AutoModel
                
        print(f"Loading DINOv3 model: {model_name}")
        
        try:
            # 从torch.hub加载DINOv3
            pretrained_model_name = "ckpts/ckpts_foundation/dinov3-vitl16-pretrain-lvd1689m"
            processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
            max_memory = {}
            #num_gpus = torch.cuda.device_count()
            #for i in range(num_gpus):
            #    gpu_memory = torch.cuda.get_device_properties(i).total_memory
            #    # 使用98%内存，只留2%作为缓冲
            #    max_memory[i] = int(gpu_memory * 0.98)
            #    gpu_memory_gb = gpu_memory / (1024**3)
            #    max_memory_gb = max_memory[i] / (1024**3)
            #    print(f"GPU {i}: Total={gpu_memory_gb:.1f}GB, Allocating={max_memory_gb:.1f}GB (98%)")
            
            model = AutoModel.from_pretrained(
                pretrained_model_name, 
                device_map="auto", 
                max_memory=max_memory if max_memory else None,  # 使用自定义内存限制
                low_cpu_mem_usage=True,  # 减少CPU内存使用
            )
            # 获取embed_dim和num_heads
            embed_dim = 1024
            print(f"✓ DINOv3 loaded: embed_dim={embed_dim}")

        except Exception as e:
            print(f"Error loading DINOv3 model: {e}")
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
        patch_size = getattr(self.model, 'patch_size', 16)
        
        # 计算需要的尺寸（向上取整到patch_size的整数倍）
        #target_H = ((H + patch_size - 1) // patch_size) * patch_size
        #target_W = ((W + patch_size - 1) // patch_size) * patch_size

        target_H = 256
        target_W = 256
        
        # 如果尺寸不匹配，进行resize
        if H != target_H or W != target_W:
            import torch.nn.functional as Fn
            # [B, C, F, H, W] → [B*F, C, H, W] → resize → [B*F, C, H', W']
            x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
            x_resized = Fn.interpolate(
                x_reshaped, 
                size=(target_H, target_W), 
                mode='bilinear', 
                align_corners=False
            )
            # [B*F, C, H', W'] → [B, F, C, H', W'] → [B, C, F, H', W']
            x = x_resized.reshape(B, F, C, target_H, target_W).permute(0, 2, 1, 3, 4)
            H, W = target_H, target_W
        
        
        # DINOv3是图像模型，需要将时间维度展开
        # [B, C, F, H, W] → [B, F, C, H, W] → [B*F, C, H, W]
        x = x.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)  # [B, F, C, H, W]
        
        # 通过DINOv3提取特征
        with torch.no_grad():
            # DINOv3 forward 返回不同格式取决于版本
            # 尝试多种方式获取patch tokens
            
            # 方法1: 使用 forward_features (推荐)
            if hasattr(self.model, 'forward_features'):
                features = self.model.forward_features(x)  # 可能返回dict或tensor
            else:
                features = self.model(pixel_values=x, output_hidden_states=True)
            
            # 处理不同的输出格式
            if hasattr(features, 'last_hidden_state'):
                features = features.last_hidden_state  # [B*F, N+1, D]
                features = features[:, 1:, :]

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
        }

# 测试代码（可选）
if __name__ == "__main__":
    # 测试输入输出格式
    adapter = DINOv3Adapter.from_config(
        resolution=256,
        checkpoint=None,
        model_name='dinov3_vitl16'
    )
    
    # 模拟输入: [B=2, C=3, F=4, H=224, W=224]
    dummy_input = torch.randn(2, 3, 4, 256, 256)
    
    print(f"Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = adapter(dummy_input)
    
    print(f"Output shape: {output.shape}")  # 应该是 [2, 4*N, D]
    print(f"Expected: [2, {4 * (256//16)**2}, {adapter.embed_dim}]")
    print(f"Feature info: {adapter.get_feature_info()}")