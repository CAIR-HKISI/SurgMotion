"""
EndoSSL Foundation Model Adapter
基于EndoSSL仓库，支持加载EndoSSL预训练模型
参考: https://github.com/royhirsch/endossl
"""
import sys
sys.path.append(".")

import torch
import torch.nn as nn
from typing import Optional
from pathlib import Path
from evals.foundation_phase_probing.modelcustom.adapters.base_adapter import BaseFoundationModelAdapter
from evals.foundation_phase_probing.modelcustom.adapters.utils import load_and_apply_checkpoint

class EndoSSLAdapter(BaseFoundationModelAdapter):
    """EndoSSL模型的Adapter - 输入格式: [B, C, F, H, W]"""
    
    def __init__(self, model, embed_dim: int, model_name: str):
        super().__init__(model, embed_dim)
        self.model_name = model_name
        self.patch_size = 16  # ViT标准patch size
    
    @classmethod
    def from_config(cls, resolution: int, checkpoint: Optional[str] = None, model_name: str = 'endossl_vitl'):
        """
        从配置创建adapter，加载EndoSSL模型
        
        Args:
            resolution: 输入分辨率（通常是224或256）
            checkpoint: checkpoint路径（如果为None，会根据model_name自动选择）
            model_name: 模型名称标识
                - 'endossl_laparo': 加载endossl_laparo_vitl权重
                - 'endossl_colono': 加载endossl_colono_vitl权重
        """
        # 根据model_name映射到对应的checkpoint路径
        checkpoint_mapping = {
            'endossl_laparo': 'ckpts/ckpts_foundation/endossl_laparo_vitl',
            'endossl_colono': 'ckpts/ckpts_foundation/endossl_colono_vitl',
            # 默认值（向后兼容）
            'endossl_vitl': 'ckpts/ckpts_foundation/endossl_laparo_vitl',
        }
        
        # 如果没有指定checkpoint，根据model_name自动选择
        if checkpoint is None:
            if model_name in checkpoint_mapping:
                checkpoint = checkpoint_mapping[model_name]
                print(f"📍 Auto-selected checkpoint for '{model_name}': {checkpoint}")
            else:
                # 如果model_name不在映射中，使用默认值
                checkpoint = checkpoint_mapping['endossl_vitl']
                print(f"⚠ Warning: Unknown model_name '{model_name}', using default checkpoint: {checkpoint}")
        
        print(f"Loading EndoSSL model: {model_name}")
        
        try:
            # EndoSSL通常使用ViT-Large架构
            # 优先使用本地ViT实现（与项目其他部分一致）
            try:
                from src.models.vision_transformer import vit_large
                model = vit_large(
                    patch_size=16,
                    img_size=resolution,
                    num_frames=1,  # 图像模型
                )
                embed_dim = 1024
                patch_size = 16
                print(f"✓ Created ViT-Large model using local implementation")
            except ImportError:
                # 备选方案：使用timm
                try:
                    import timm
                    model = timm.create_model(
                        'vit_large_patch16_224',
                        pretrained=False,
                        num_classes=0,  # 移除分类头
                        img_size=resolution
                    )
                    embed_dim = 1024
                    patch_size = 16
                    print(f"✓ Created ViT-Large model using timm")
                except ImportError:
                    # 最后备选：使用DINOv2架构
                    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=False)
                    embed_dim = 1024
                    patch_size = 14
                    print(f"✓ Created ViT-Large model using DINOv2 architecture")
            
            print(f"  - embed_dim: {embed_dim}")
            print(f"  - patch_size: {patch_size}")
            print(f"  - resolution: {resolution}")
            
            # 检查checkpoint是文件还是目录
            checkpoint_path_obj = Path(checkpoint)
            if checkpoint_path_obj.is_dir():
                # 如果是目录，查找checkpoint文件
                checkpoint_files = list(checkpoint_path_obj.glob("*.pth")) + \
                                 list(checkpoint_path_obj.glob("*.pt")) + \
                                 list(checkpoint_path_obj.glob("*.ckpt"))
                if checkpoint_files:
                    checkpoint_path = str(checkpoint_files[0])
                    print(f"  Found checkpoint file: {checkpoint_path}")
                else:
                    print(f"  Warning: No checkpoint file found in directory: {checkpoint_path}")
                    checkpoint_path = None
            elif not checkpoint_path_obj.exists():
                print(f"  Warning: Checkpoint not found: {checkpoint_path}")
                checkpoint_path = None
            else:
                checkpoint_path = checkpoint
            
            # 加载checkpoint
            if checkpoint_path:
                # 尝试不同的前缀移除策略
                prefixes_to_try = [None, 'model.', 'backbone.', 'encoder.', 'module.', 'teacher.', 'student.']
                
                loaded = False
                for prefix in prefixes_to_try:
                    success, info = load_and_apply_checkpoint(
                        model=model,
                        checkpoint_path=checkpoint_path,
                        default_path=None,
                        strict=False,
                        key_prefix_to_remove=prefix,
                        verbose=(prefix is None)  # 只在第一次尝试时详细输出
                    )
                    
                    if success:
                        if prefix:
                            print(f"  ✓ Successfully loaded with prefix '{prefix}' removed")
                        else:
                            print(f"  ✓ Successfully loaded checkpoint")
                        loaded = True
                        break
                
                if not loaded:
                    print(f"  Warning: Failed to load checkpoint: {info}")
            else:
                print("  Using randomly initialized weights (no checkpoint loaded)")
        
        except Exception as e:
            print(f"Error loading EndoSSL model: {e}")
            import traceback
            traceback.print_exc()
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
        
        # 调整分辨率到模型期望的尺寸（通常是224）
        target_H, target_W = 224, 224
        
        # 如果尺寸不匹配，进行resize
        if H != target_H or W != target_W:
            import torch.nn.functional as Fn
            # [B, C, F, H, W] → [B*F, C, H, W] → resize → [B*F, C, 224, 224]
            x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
            x_resized = Fn.interpolate(
                x_reshaped, 
                size=(target_H, target_W), 
                mode='bilinear', 
                align_corners=False
            )
            # [B*F, C, 224, 224] → [B, F, C, 224, 224] → [B, C, F, 224, 224]
            x = x_resized.reshape(B, F, C, target_H, target_W).permute(0, 2, 1, 3, 4)
            H, W = target_H, target_W
        
        # EndoSSL是图像模型，需要将时间维度展开
        # [B, C, F, H, W] → [B, F, C, H, W] → [B*F, C, H, W]
        x = x.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
        x = x.reshape(B * F, C, H, W)  # [B*F, C, H, W]
        
        # 通过EndoSSL提取特征
        with torch.no_grad():
            # 根据模型类型提取特征
            if hasattr(self.model, 'forward_features'):
                # timm或自定义ViT模型
                features = self.model.forward_features(x)
            elif hasattr(self.model, 'get_intermediate_layers'):
                # DINOv2风格
                output = self.model.get_intermediate_layers(x, n=1)
                features = output[0]  # [B*F, N+1, D]
            elif hasattr(self.model, 'forward'):
                # 标准ViT forward
                # 尝试获取patch tokens
                output = self.model(x)
                if isinstance(output, torch.Tensor):
                    features = output
                else:
                    # 如果返回tuple或dict，取第一个
                    features = output[0] if isinstance(output, (tuple, list)) else output
            else:
                raise ValueError("Cannot determine how to extract features from EndoSSL model")
            
            # 处理不同的输出格式
            if features.dim() == 3:
                # [B*F, N+1, D] 或 [B*F, N, D]
                # 检查是否包含CLS token（通常在第一个位置）
                if features.shape[1] == (H // self.patch_size) * (W // self.patch_size) + 1:
                    # 包含CLS token，去掉它
                    features = features[:, 1:, :]  # [B*F, N, D]
            elif features.dim() == 2:
                # [B*F, D] - 只有全局特征，需要重新处理
                raise ValueError(f"EndoSSL returned global features only, expected patch tokens. Shape: {features.shape}")
            else:
                raise ValueError(f"Unexpected feature shape: {features.shape}")
        
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
            'patch_size': self.patch_size,
            'architecture': 'ViT-Large',
            'pretrain': 'EndoSSL',
        }


# 测试代码
if __name__ == "__main__":
    """
    测试EndoSSL adapter的输入输出格式
    
    Expected:
        Input: [B=2, C=3, F=64, H=224, W=224]
        Output: [2, 64*196, 1024] = [2, 12544, 1024]
    """
    # 测试输入输出格式
    adapter = EndoSSLAdapter.from_config(
        resolution=224,
        checkpoint="ckpts/ckpts_foundation/endossl_colono_vitl",
        model_name='endossl_vitl'
    )
    
    # 模拟输入: [B=2, C=3, F=4, H=224, W=224]
    dummy_input = torch.randn(2, 3, 4, 224, 224)
    
    print(f"\nInput shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = adapter(dummy_input)
    
    print(f"Output shape: {output.shape}")  # 应该是 [2, 4*196, 1024]
    print(f"Expected: [2, {4 * (224//16)**2}, {adapter.embed_dim}]")
    print(f"\nFeature info: {adapter.get_feature_info()}")