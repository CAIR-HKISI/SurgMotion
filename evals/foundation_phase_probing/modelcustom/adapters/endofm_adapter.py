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
from evals.foundation_phase_probing.modelcustom.adapters.utils import load_and_apply_checkpoint, parse_args, load_config


class EndoFMAdapter(BaseFoundationModelAdapter):
    """EndoViT模型的Adapter - 输入格式: [B, C, F, H, W]"""
    
    def __init__(self, model, embed_dim: int, model_name: str):
        super().__init__(model, embed_dim)
        self.model_name = model_name
        self.patch_size = 16
    
    @classmethod
    def from_config(cls, resolution: int, frames_per_clip: int, checkpoint: Optional[str] = None, model_name: str = ''):
        """
        从配置创建adapter，使用官方EndoViT定义
        
        Args:
            resolution: 输入分辨率（通常是224）
            checkpoint: checkpoint路径，默认为EndoViT_SPR
            model_name: 模型架构名称 ('vit_base_patch16', 'vit_large_patch16', 'vit_huge_patch14')
        """
        import sys
        
        # 添加EndoViT路径到sys.path
        endofm_path = Path(__file__).parent.parent.parent.parent.parent / "foundation_models" / "Endo-FM"
        if str(endofm_path) not in sys.path:
            sys.path.insert(0, str(endofm_path))
        
        # 导入官方的模型定义和配置工具
        from models import get_vit_base_patch16_224

        print(f"Loading Endo-FM model: {model_name}")
        
        try:
            # 创建配置对象（模拟argparse）
            class SimpleArgs:
                def __init__(self):
                    self.cfg_file = str(endofm_path / "models" / "configs" / "Kinetics" / "TimeSformer_divST_8x32_224.yaml")
                    self.opts = None
                    self.num_shards = 1
                    self.shard_id = 0
            
            # 加载Endo-FM配置
            args = SimpleArgs()
            config = load_config(args)
            
            # 修改配置以适应我们的设置
            config.DATA.TRAIN_CROP_SIZE = resolution
            config.DATA.NUM_FRAMES = frames_per_clip
            config.MODEL.NUM_CLASSES = 400  # 默认值，后面会移除head
            config.TIMESFORMER.ATTENTION_TYPE = 'divided_space_time'  # Endo-FM的标准注意力类型
            
            # 创建模型（no_head=True 表示不使用分类头）
            model = get_vit_base_patch16_224(cfg=config, no_head=True)
            embed_dim = 768  # ViT-Base的embed_dim
            patch_size = 16
            
            print(f"✓ Endo-FM architecture created:")
            print(f"  - embed_dim: {embed_dim}")
            print(f"  - patch_size: {patch_size}")
            print(f"  - attention_type: {config.TIMESFORMER.ATTENTION_TYPE}")
            print(f"  - num_frames: {frames_per_clip}")
            print(f"  - resolution: {resolution}")
            
            # 使用通用工具函数加载checkpoint
            success, info = load_and_apply_checkpoint(
                model=model,
                checkpoint_path=checkpoint,
                default_path="ckpts/ckpts_foundation/endo_fm.pth",
                strict=False,
                key_prefix_to_remove="backbone.",  # Endo-FM权重带有 'backbone.' 前缀
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

        # 通过Endo-FM提取特征
        with torch.no_grad():
            # 使用官方的forward_features方法
            # pool_type='no_pooling' 返回 [B*F, N, D] (不包括CLS token)
            features = self.model.forward_features(x, get_all=True)
        
        # 验证输出形状
        if features.dim() != 3:
            raise ValueError(f"Expected 3D features [B*F, N, D], got shape: {features.shape}")
        #print(f"Features shape: {features.shape}")
        features = features[:, 1:, :]
        #print(f"Features shape after removing CLS token: {features.shape}")

        B, N, D = features.shape
        
        #print(f"📊 Endo-FM feature shape: {features.shape}")
        #print(f"   - Expected N_total: {F * (H//16) * (W//16)} (F={F}, spatial_patches={(H//16)*(W//16)})")
        
        
        return features

    def get_feature_info(self):
        """返回特征提取的信息（用于调试）"""
        return {
            'model_name': self.model_name,
            'embed_dim': self.embed_dim,
            'patch_size': 16,
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
    adapter = EndoFMAdapter.from_config(
        resolution=256,
        frames_per_clip=64,
        checkpoint="ckpts/ckpts_foundation/endofm_cholec80.pth",
        model_name='endofm'
    )
    
    # 模拟输入: [B=2, C=3, F=4, H=224, W=224]
    dummy_input = torch.randn(2, 3, 4, 224, 224)
    
    print(f"Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = adapter(dummy_input)
    
    print(f"Output shape: {output.shape}")  # 应该是 [2, 4*N, D]
    print(f"Expected: [2, {4 * (224//16)**2}, {adapter.embed_dim}]")
    print(f"Feature info: {adapter.get_feature_info()}")