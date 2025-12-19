"""
SurgVISTA Foundation Model Adapter
基于官方SurgVISTA仓库定义，支持加载SurgVISTA checkpoint
仿照 EndoViT adapter 的设计
"""

import torch
import torch.nn as nn
from typing import Optional
from pathlib import Path
from functools import partial
from base_adapter import BaseFoundationModelAdapter
from utils import load_and_apply_checkpoint


class SurgVISTAAdapter(BaseFoundationModelAdapter):
    """SurgVISTA模型的Adapter - 输入格式: [B, C, F, H, W]"""
    
    def __init__(self, model, embed_dim: int, model_name: str):
        super().__init__(model, embed_dim)
        self.model_name = model_name
    
    @classmethod
    def from_config(cls, resolution: int, checkpoint: Optional[str] = None, model_name: str = 'vit_base_patch16'):
        """
        从配置创建adapter，使用官方SurgVISTA定义
        
        Args:
            resolution: 输入分辨率（通常是224）
            checkpoint: checkpoint路径，默认为SurgVISTA预训练权重
            model_name: 模型架构名称 ('vit_base_patch16', 'vit_large_patch16')
        """
        import sys
        
        # 添加SurgVISTA路径到sys.path
        surgvista_path = Path(__file__).parent.parent.parent.parent.parent / "foundation_models" / "SurgVISTA" / "downstream"
        if str(surgvista_path) not in sys.path:
            sys.path.insert(0, str(surgvista_path))
        
        # 导入官方的模型定义和工具
        from model.unifiedmodel import unified_base
        import utils
        
        print(f"Loading SurgVISTA model: {model_name}")
        
        try:
            # 创建模型（不使用mean pooling，保留所有tokens）
            model = unified_base(
                pretrained=False,
                pretrain_path=None,
                num_classes=0,  # 不需要分类头
                use_mean_pooling=False,  # 关键：保留所有tokens
                all_frames=128,
                tubelet_size=2,
            )
            
            embed_dim = 768
            patch_size = 16
            
            print(f"SurgVISTA model created: embed_dim={embed_dim}, patch_size={patch_size}")
            
            # 加载checkpoint
            success, info = load_and_apply_checkpoint(
                model=model,
                checkpoint_path=checkpoint,
                default_path="/home/chen_chuxi/NSJepa/ckpts_foundation/vit_large_patch16_224_surgery.pth",
                strict=False,
                key_prefix_to_remove=None,
                verbose=True
            )
            
            if not success:
                print(f"Warning: {info}")

        except Exception as e:
            print(f"Error loading SurgVISTA model: {e}")
            raise e
        
        return cls(model, embed_dim, model_name)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, F, H, W] 视频输入
        
        Returns:
            features: [B, F*N, D] 所有帧的所有patch tokens
        """
        B, C, F, H, W = x.shape

        # Resize to 224x224
        target_H, target_W = 224, 224
        if H != target_H or W != target_W:
            import torch.nn.functional as fn
            x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
            x_resized = fn.interpolate(x_reshaped, size=(target_H, target_W), 
                                       mode='bilinear', align_corners=False)
            x = x_resized.reshape(B, F, C, target_H, target_W).permute(0, 2, 1, 3, 4)
            H, W = target_H, target_W
        
        # 确保帧数匹配（SurgVISTA通常使用16帧）
        #expected_frames = 16
        #if F != expected_frames:
        #    if F < expected_frames:
        #        padding = expected_frames - F
        #        last_frame = x[:, :, -1:, :, :].repeat(1, 1, padding, 1, 1)
        #        x = torch.cat([x, last_frame], dim=2)
        #    else:
        #        x = x[:, :, :expected_frames, :, :]
       #     F = expected_frames
        
        # 提取所有patch tokens
        with torch.no_grad():
            # Patch embedding
            x_tokens = self.model.patch_embed(x)  # [B, N, D]
            
            # Position embedding
            if self.model.pos_embed is not None:
                B_tokens, N_tokens, D_tokens = x_tokens.shape
                if isinstance(self.model.pos_embed, nn.Parameter):
                    # 可学习的位置编码：直接使用
                    x_tokens = x_tokens + self.model.pos_embed.expand(B_tokens, -1, -1).type_as(x_tokens).to(x_tokens.device).clone().detach()
                else:
                    # 正弦位置编码：需要处理帧数不匹配的情况
                    # 如果输入帧数与位置编码不匹配，进行插值
                    pos_embed = self.model.pos_embed
                    if isinstance(pos_embed, torch.Tensor):
                        pos_embed_tensor = pos_embed
                    else:
                        # numpy array
                        pos_embed_tensor = torch.from_numpy(pos_embed).to(x_tokens.device).type_as(x_tokens)
                    
                    # 检查位置编码长度是否匹配
                    if pos_embed_tensor.shape[1] != N_tokens:
                        # 位置编码长度不匹配，进行插值
                        # pos_embed: [1, N_old, D] -> [1, N_new, D]
                        import torch.nn.functional as F
                        pos_embed_tensor = F.interpolate(
                            pos_embed_tensor.transpose(1, 2),  # [1, D, N_old]
                            size=N_tokens,  # 插值到 N_tokens
                            mode='linear',
                            align_corners=False
                        ).transpose(1, 2)  # [1, N_new, D]
                    
                    x_tokens = x_tokens + pos_embed_tensor.expand(B_tokens, -1, -1)
            
            x_tokens = self.model.pos_drop(x_tokens)
            
            # Transformer blocks
            for blk in self.model.blocks:
                x_tokens = blk(x_tokens)
            
            # Final norm
            features = self.model.norm(x_tokens)  # [B, N, D]
        
        # features已经是 [B, N, D] 格式，其中N包含所有帧的所有patches
        # 与其他adapter保持一致，直接返回
        return features
    
    def get_feature_info(self):
        """返回特征提取的信息"""
        return {
            'model_name': self.model_name,
            'embed_dim': self.embed_dim,
            'patch_size': 16,
            'tubelet_size': getattr(self.model.patch_embed, 'tubelet_size', 2),
            'max_frames': 128,  # 与其他模型保持一致
        }


if __name__ == "__main__":
    adapter = SurgVISTAAdapter.from_config(
        resolution=224,
        checkpoint="/home/chen_chuxi/NSJepa/ckpts_foundation/vit_large_patch16_224_surgery.pth",
        model_name='vit_base_patch16'
    )
    
    dummy_input = torch.randn(2, 3, 16, 224, 224)
    print(f"Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = adapter(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Feature info: {adapter.get_feature_info()}")