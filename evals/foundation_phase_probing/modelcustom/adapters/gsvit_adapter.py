"""
GSViT Foundation Model Adapter
基于GSViT (General Surgery Vision Transformer) 模型
参考: https://github.com/royhirsch/GSViT
"""
import sys
sys.path.append(".")

import torch
import torch.nn as nn
from typing import Optional
from pathlib import Path
from evals.foundation_phase_probing.modelcustom.adapters.base_adapter import BaseFoundationModelAdapter
from evals.foundation_phase_probing.modelcustom.adapters.utils import load_and_apply_checkpoint


class GSViTAdapter(BaseFoundationModelAdapter):
    """GSViT模型的Adapter - 输入格式: [B, C, F, H, W]"""
    
    def __init__(self, model, embed_dim: int, model_name: str):
        super().__init__(model, embed_dim)
        self.model_name = model_name
        self.patch_size = 16  # EfficientViT标准patch size
    
    @classmethod
    def from_config(cls, resolution: int, checkpoint: Optional[str] = None, model_name: str = 'gsvit'):
        """
        从配置创建adapter，加载GSViT模型
        
        Args:
            resolution: 输入分辨率（通常是224）
            checkpoint: checkpoint路径
            model_name: 模型名称标识
        """
        import sys
        
        # 添加GSViT路径到sys.path
        gsvit_path = Path(__file__).parent.parent.parent.parent.parent / "foundation_models" / "GSViT"
        if str(gsvit_path) not in sys.path:
            sys.path.insert(0, str(gsvit_path))
        
        print(f"Loading GSViT model: {model_name}")
        
        try:
            # 导入GSViT模型定义
            from EfficientViT.classification.model.build import EfficientViT_M5
            
            # 创建EfficientViT_M5模型（不使用分类头）
            # GSViT基于EfficientViT_M5，移除分类头后只保留特征提取部分
            evit_model = EfficientViT_M5(pretrained=False, num_classes=0)
            # 移除分类头（如果还有的话）
            if hasattr(evit_model, 'head'):
                evit_model.head = nn.Identity()
            
            # EfficientViT_M5的最后一层embed_dim是384
            embed_dim = 384
            patch_size = 16
            
            print(f"✓ GSViT architecture created:")
            print(f"  - embed_dim: {embed_dim}")
            print(f"  - patch_size: {patch_size}")
            print(f"  - resolution: {resolution}")
            
            # 加载checkpoint
            checkpoint_path = checkpoint or "ckpts/ckpts_foundation/GSViT.pkl"
            
            if not Path(checkpoint_path).exists():
                print(f"  Warning: Checkpoint not found: {checkpoint_path}")
                checkpoint_path = None
            
            if checkpoint_path:
                print(f"  Loading checkpoint from: {checkpoint_path}")
                
                # GSViT的checkpoint格式：包含整个EfficientViT模型（已移除分类头）
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                    
                    # 处理不同的checkpoint格式
                    if isinstance(checkpoint, dict):
                        # 检查是否有特定的key
                        if 'evit' in checkpoint:
                            state_dict = checkpoint['evit']
                        elif 'model' in checkpoint:
                            state_dict = checkpoint['model']
                        elif 'state_dict' in checkpoint:
                            state_dict = checkpoint['state_dict']
                        else:
                            state_dict = checkpoint
                    else:
                        state_dict = checkpoint
                    
                    # 尝试加载权重
                    # GSViT的checkpoint可能包含完整的EfficientViT模型（已移除head）
                    # 需要匹配模型结构
                    try:
                        # 尝试直接加载
                        msg = evit_model.load_state_dict(state_dict, strict=False)
                        print(f"  ✓ Checkpoint loaded")
                        if msg.missing_keys:
                            print(f"    Missing keys: {len(msg.missing_keys)}")
                        if msg.unexpected_keys:
                            print(f"    Unexpected keys: {len(msg.unexpected_keys)}")
                    except Exception as e1:
                        # 如果直接加载失败，尝试移除前缀
                        print(f"  Trying to load with prefix removal...")
                        cleaned_state_dict = {}
                        for k, v in state_dict.items():
                            # 尝试移除常见前缀
                            new_key = k
                            for prefix in ['evit.', 'model.', 'backbone.', 'encoder.', 'module.']:
                                if new_key.startswith(prefix):
                                    new_key = new_key[len(prefix):]
                                    break
                            cleaned_state_dict[new_key] = v
                        
                        msg = evit_model.load_state_dict(cleaned_state_dict, strict=False)
                        print(f"  ✓ Checkpoint loaded with prefix removal")
                        if msg.missing_keys:
                            print(f"    Missing keys: {len(msg.missing_keys)}")
                        if msg.unexpected_keys:
                            print(f"    Unexpected keys: {len(msg.unexpected_keys)}")
                
                except Exception as e:
                    print(f"  Warning: Failed to load checkpoint: {e}")
                    print(f"  Using randomly initialized weights")
            else:
                print("  Using randomly initialized weights (no checkpoint loaded)")
            
            # 包装模型以处理输入（颜色通道翻转）
            model = GSViTWrapper(evit_model)
        
        except Exception as e:
            print(f"Error loading GSViT model: {e}")
            import traceback
            traceback.print_exc()
            raise e
        
        return cls(model, embed_dim, model_name)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, F, H, W] 视频输入
        Returns:
            features: [B, F*N, D] 
        """
        B, C, F, H, W = x.shape
        
        # 调整分辨率到224（EfficientViT标准输入）
        target_H, target_W = 224, 224
        
        if H != target_H or W != target_W:
            import torch.nn.functional as Fn
            x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
            x_resized = Fn.interpolate(
                x_reshaped, 
                size=(target_H, target_W), 
                mode='bilinear', 
                align_corners=False
            )
            x = x_resized.reshape(B, F, C, target_H, target_W).permute(0, 2, 1, 3, 4)
            H, W = target_H, target_W
        
        # 展开时间维度: [B, C, F, H, W] -> [B*F, C, H, W]
        x = x.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
        
        # 通过GSViT提取特征
        with torch.no_grad():
            # GSViT输出特征图 [B*F, 384, 4, 4]
            feature_map = self.model(x)
        
        # 将特征图转换为patch tokens格式
        # [B*F, 384, 4, 4] -> [B*F, 16, 384]
        BF, C_feat, H_feat, W_feat = feature_map.shape
        features = feature_map.flatten(2)  # [B*F, 384, 16]
        features = features.permute(0, 2, 1)  # [B*F, 16, 384]
        
        # 验证输出形状
        if features.dim() != 3:
            raise ValueError(f"Expected 3D features [B*F, N, D], got: {features.shape}")
        
        BF_check, N, D = features.shape
        assert BF_check == B * F, f"Shape mismatch: {BF_check} != {B} * {F}"
        
        # 重塑为 [B, F*N, D]
        features = features.reshape(B, F, N, D)
        features = features.reshape(B, F * N, D)
        
        return features
    
    def get_feature_info(self):
        """返回特征信息"""
        return {
            'model_name': self.model_name,
            'embed_dim': self.embed_dim,
            'patch_size': self.patch_size,
            'architecture': 'EfficientViT-M5',
            'pretrain': 'GSViT',
        }


class GSViTWrapper(nn.Module):
    """
    GSViT模型包装器，处理输入的颜色通道翻转
    """
    def __init__(self, evit_model):
        super().__init__()
        self.evit = evit_model
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 输入图像（RGB格式）
        Returns:
            feature_map: [B, 384, 4, 4] 特征图
        """
        # GSViT需要翻转颜色通道（RGB -> BGR）
        # 交换R和B通道
        x_flipped = x.clone()
        x_flipped[:, 0, :, :] = x[:, 2, :, :]  # R <- B
        x_flipped[:, 2, :, :] = x[:, 0, :, :]  # B <- R
        
        # 通过EfficientViT提取特征
        # 注意：需要修改forward以返回特征图而不是全局池化后的特征
        x = self.evit.patch_embed(x_flipped)
        x = self.evit.blocks1(x)
        x = self.evit.blocks2(x)
        x = self.evit.blocks3(x)
        # 不进行全局池化，返回特征图 [B, 384, 4, 4]
        return x


# 测试代码
if __name__ == "__main__":
    adapter = GSViTAdapter.from_config(
        resolution=224,
        checkpoint="ckpts/ckpts_foundation/GSViT.pkl",
        model_name='gsvit'
    )
    
    dummy_input = torch.randn(2, 3, 4, 224, 224)
    print(f"Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = adapter(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected: [2, {4 * 16}, {adapter.embed_dim}]")  # 4 frames * 16 patches = 64
    print(f"Feature info: {adapter.get_feature_info()}")