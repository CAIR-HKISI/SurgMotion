"""
VideoMAE/VideoMAEv2 Foundation Model Adapter
支持通过 HuggingFace Transformers 加载 VideoMAEv2 模型

长窗口支持：
- 标准 VideoMAE 使用 16 帧输入
- 本 adapter 支持 64 帧长窗口，通过分段处理实现
- 将 64 帧分成 4 个 16 帧片段，分别提取特征后拼接
"""

import torch
import torch.nn as nn
import torch.nn.functional as Fn
from typing import Optional, List
from .base_adapter import BaseFoundationModelAdapter


class VideoMAEAdapter(BaseFoundationModelAdapter):
    """
    VideoMAE/VideoMAEv2 模型的 Adapter - 支持 64 帧长窗口
    
    输入格式: [B, C, F, H, W]  (F=64 帧)
    输出格式: [B, N, D] 其中 N = num_segments * tokens_per_segment
    
    处理策略：
        - 将 64 帧分成 4 个 16 帧片段
        - 每个片段独立通过 VideoMAE 提取特征
        - 拼接所有片段的特征
    
    支持的模型:
        - OpenGVLab/VideoMAEv2-Base (768D)
        - OpenGVLab/VideoMAEv2-Large (1024D)
        - OpenGVLab/VideoMAEv2-Huge (1280D)
        - OpenGVLab/VideoMAEv2-Giant (1408D)
    """
    
    # 模型配置
    MODEL_CONFIGS = {
        'videomaev2_base': {
            'pretrained_name': 'ckpts/ckpts_foundation/OpenGVLab/VideoMAEv2-Base',
            'embed_dim': 768,
            'patch_size': 16,
            'tubelet_size': 2,
            'native_frames': 16,  # VideoMAE 原生支持的帧数
        },
        'videomaev2_large': {
            'pretrained_name': 'ckpts/ckpts_foundation/OpenGVLab/VideoMAEv2-Large',
            'embed_dim': 1024,
            'patch_size': 16,
            'tubelet_size': 2,
            'native_frames': 16,
        },
        'videomaev2_huge': {
            'pretrained_name': 'ckpts/ckpts_foundation/OpenGVLab/VideoMAEv2-Huge',
            'embed_dim': 1280,
            'patch_size': 16,
            'tubelet_size': 2,
            'native_frames': 16,
        },
        'videomaev2_giant': {
            'pretrained_name': 'ckpts/ckpts_foundation/OpenGVLab/VideoMAEv2-Giant',
            'embed_dim': 1408,
            'patch_size': 14,
            'tubelet_size': 2,
            'native_frames': 16,
        },
    }
    
    def __init__(
        self, 
        model, 
        embed_dim: int, 
        model_name: str,
        processor=None,
        tubelet_size: int = 2,
        patch_size: int = 16,
        native_frames: int = 16,  # VideoMAE 原生帧数
        target_frames: int = 64,  # 目标输入帧数（长窗口）
    ):
        super().__init__(model, embed_dim)
        self.model_name = model_name
        self.processor = processor
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.native_frames = native_frames
        self.target_frames = target_frames
        
        # 计算分段数量
        self.num_segments = target_frames // native_frames
        if target_frames % native_frames != 0:
            print(f"⚠️ Warning: target_frames ({target_frames}) is not divisible by native_frames ({native_frames})")
            self.num_segments = (target_frames + native_frames - 1) // native_frames
        
        print(f"📊 VideoMAE Long-Window Config:")
        print(f"   - target_frames: {target_frames}")
        print(f"   - native_frames: {native_frames}")
        print(f"   - num_segments: {self.num_segments}")
    
    @classmethod
    def from_config(
        cls, 
        resolution: int, 
        frames_per_clip: int = 64,  # 默认 64 帧长窗口
        checkpoint: Optional[str] = None, 
        model_name: str = 'videomaev2_large'
    ):
        """
        从配置创建 adapter
        
        Args:
            resolution: 输入分辨率（通常是 224）
            frames_per_clip: 每个 clip 的帧数（默认 64 帧长窗口）
            checkpoint: 自定义 checkpoint 路径（可选）
            model_name: 模型名称，支持:
                - 'videomaev2_base', 'videomaev2_large', 'videomaev2_huge', 'videomaev2_giant'
        """
        from transformers import VideoMAEImageProcessor, AutoModel, AutoConfig
        
        print(f"Loading VideoMAE model: {model_name}")
        
        # 获取模型配置
        if model_name in cls.MODEL_CONFIGS:
            config_info = cls.MODEL_CONFIGS[model_name]
            pretrained_name = config_info['pretrained_name']
            embed_dim = config_info['embed_dim']
            patch_size = config_info['patch_size']
            tubelet_size = config_info['tubelet_size']
            native_frames = config_info['native_frames']
        else:
            # 允许直接使用 HuggingFace 模型名
            pretrained_name = model_name
            embed_dim = 1024
            patch_size = 16
            tubelet_size = 2
            native_frames = 16
        
        try:
            # 加载配置
            config = AutoConfig.from_pretrained(pretrained_name, trust_remote_code=True)
            
            # 更新 embed_dim（从配置中获取更准确的值）
            if hasattr(config, 'hidden_size'):
                embed_dim = config.hidden_size
            
            # 加载 processor
            processor = VideoMAEImageProcessor.from_pretrained(pretrained_name)
            
            # 加载模型
            if checkpoint and checkpoint.lower() != 'none':
                print(f"Loading from custom checkpoint: {checkpoint}")
                model = AutoModel.from_pretrained(
                    pretrained_name, 
                    config=config, 
                    trust_remote_code=True
                )
                # 加载自定义权重
                state_dict = torch.load(checkpoint, map_location='cpu')
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                elif 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                
                # 移除可能的前缀
                for prefix in ['module.', 'backbone.', 'encoder.']:
                    state_dict = {k.replace(prefix, ''): v for k, v in state_dict.items()}
                
                model.load_state_dict(state_dict, strict=False)
                print(f"✓ Loaded custom checkpoint")
            else:
                model = AutoModel.from_pretrained(
                    pretrained_name, 
                    config=config, 
                    trust_remote_code=True
                )
            
            print(f"✓ VideoMAE loaded successfully:")
            print(f"  - pretrained: {pretrained_name}")
            print(f"  - embed_dim: {embed_dim}")
            print(f"  - patch_size: {patch_size}")
            print(f"  - tubelet_size: {tubelet_size}")
            print(f"  - native_frames: {native_frames}")
            print(f"  - target_frames (long window): {frames_per_clip}")
            
        except Exception as e:
            print(f"Error loading VideoMAE model: {e}")
            raise e
        
        return cls(
            model=model, 
            embed_dim=embed_dim, 
            model_name=model_name,
            processor=processor,
            tubelet_size=tubelet_size,
            patch_size=patch_size,
            native_frames=native_frames,
            target_frames=frames_per_clip,
        )
    
    def _adjust_temporal_length(self, x: torch.Tensor) -> torch.Tensor:
        """
        调整时间长度到 target_frames
        
        Args:
            x: [B, C, F, H, W]
        Returns:
            x: [B, C, target_frames, H, W]
        """
        B, C, F, H, W = x.shape
        
        if F == self.target_frames:
            return x
        
        if F > self.target_frames:
            # 均匀采样
            indices = torch.linspace(0, F - 1, self.target_frames).long()
            x = x[:, :, indices, :, :]
        else:
            # 重复填充
            repeat_times = (self.target_frames + F - 1) // F
            x = x.repeat(1, 1, repeat_times, 1, 1)[:, :, :self.target_frames, :, :]
        
        return x
    
    def _adjust_spatial_size(self, x: torch.Tensor, target_size: int = 224) -> torch.Tensor:
        """
        调整空间分辨率
        
        Args:
            x: [B, C, F, H, W]
            target_size: 目标分辨率
        Returns:
            x: [B, C, F, target_size, target_size]
        """
        B, C, F, H, W = x.shape
        
        if H == target_size and W == target_size:
            return x
        
        # [B, C, F, H, W] -> [B*F, C, H, W]
        x = x.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
        x = Fn.interpolate(x, size=(target_size, target_size), mode='bilinear', align_corners=False)
        # [B*F, C, H', W'] -> [B, F, C, H', W'] -> [B, C, F, H', W']
        x = x.reshape(B, F, C, target_size, target_size).permute(0, 2, 1, 3, 4)
        
        return x
    
    def _split_into_segments(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        将长视频分成多个片段
        
        Args:
            x: [B, C, F, H, W] 其中 F = target_frames (64)
        Returns:
            segments: List of [B, C, native_frames, H, W] (4 个片段)
        """
        B, C, F, H, W = x.shape
        segments = []
        
        for i in range(self.num_segments):
            start_idx = i * self.native_frames
            end_idx = min(start_idx + self.native_frames, F)
            
            segment = x[:, :, start_idx:end_idx, :, :]
            
            # 如果最后一个片段不足 native_frames，进行填充
            if segment.shape[2] < self.native_frames:
                pad_size = self.native_frames - segment.shape[2]
                # 重复最后一帧填充
                padding = segment[:, :, -1:, :, :].repeat(1, 1, pad_size, 1, 1)
                segment = torch.cat([segment, padding], dim=2)
            
            segments.append(segment)
        
        return segments
    
    def _extract_segment_features(self, segment: torch.Tensor) -> torch.Tensor:
        """
        提取单个片段的特征（patch tokens，不是 pooled）
        
        Args:
            segment: [B, C, native_frames, H, W]
        Returns:
            features: [B, N, D] 其中 N = (native_frames/tubelet) * (H/patch) * (W/patch)
        """
        # VideoMAEv2 (OpenGVLab) 默认使用 mean pooling 返回 pooled 特征
        # 我们需要获取 patch tokens，直接访问内部 VisionTransformer
        
        # 获取内部 VisionTransformer 模型
        vit_model = self._get_inner_vit()
        
        # 手动提取 patch tokens（绕过 mean pooling）
        features = self._manual_extract_patch_tokens(vit_model, segment)
        
        # 确保是 3D 张量 [B, N, D]
        if features.dim() == 2:
            features = features.unsqueeze(1)
        elif features.dim() != 3:
            raise ValueError(f"Expected 2D or 3D features, got shape: {features.shape}")
        
        return features
    
    def _get_inner_vit(self):
        """获取内部的 VisionTransformer 模型"""
        # OpenGVLab VideoMAEv2 结构: VideoMAEv2 -> model (VisionTransformer)
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'patch_embed'):
            return self.model.model
        elif hasattr(self.model, 'patch_embed'):
            return self.model
        elif hasattr(self.model, 'videomae'):
            return self.model.videomae
        elif hasattr(self.model, 'encoder'):
            return self.model.encoder
        else:
            raise ValueError("Cannot find inner VisionTransformer")
    
    def _manual_extract_patch_tokens(self, vit: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        手动提取 VideoMAEv2 的 patch tokens（绕过 mean pooling）
        
        模型结构:
            patch_embed -> pos_drop -> blocks -> fc_norm
        
        Args:
            vit: VisionTransformer 模型
            x: [B, C, T, H, W]
        Returns:
            features: [B, N, D] 所有 patch tokens
        """
        B = x.shape[0]
        
        # 1. Patch embedding: [B, C, T, H, W] -> [B, N, D]
        x = vit.patch_embed(x)
        
        # 2. 添加位置编码（如果使用 learnable pos embed）
        if hasattr(vit, 'pos_embed') and vit.pos_embed is not None:
            # 确保 pos_embed 在正确的设备上
            if vit.pos_embed.device != x.device:
                x = x + vit.pos_embed.to(x.device)
            else:
                x = x + vit.pos_embed
        
        # 3. Position dropout
        if hasattr(vit, 'pos_drop'):
            x = vit.pos_drop(x)
        
        # 4. 通过所有 Transformer blocks
        for blk in vit.blocks:
            x = blk(x)
        
        # 5. Final norm (fc_norm 而不是 mean pooling)
        # 注意：不使用 mean pooling，保留所有 patch tokens
        if hasattr(vit, 'fc_norm') and vit.fc_norm is not None:
            x = vit.fc_norm(x)
        elif hasattr(vit, 'norm') and vit.norm is not None:
            x = vit.norm(x)
        
        return x  # [B, N, D]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 支持 64 帧长窗口
        
        Args:
            x: [B, C, F, H, W] 视频输入
               - B: batch size
               - C: channels (通常是 3)
               - F: frames (64 帧长窗口)
               - H, W: 高度和宽度
        
        Returns:
            features: [B, N_total, D] 
               - N_total = num_segments * N_per_segment
               - N_per_segment = (native_frames/tubelet) * (H/patch) * (W/patch)
               - D: embed_dim
        """
        B, C, F, H, W = x.shape
        
        # 1. 调整时间长度到 target_frames (64)
        x = self._adjust_temporal_length(x)
        
        # 2. 调整空间分辨率到 224x224
        target_spatial = 224
        if H != target_spatial or W != target_spatial:
            x = self._adjust_spatial_size(x, target_spatial)
            H, W = target_spatial, target_spatial
        
        # 3. 分段处理
        segments = self._split_into_segments(x)  # List of [B, C, 16, H, W]
        
        # 4. 提取每个片段的特征
        all_features = []
        with torch.no_grad():
            for i, segment in enumerate(segments):
                features = self._extract_segment_features(segment)  # [B, N, D]
                all_features.append(features)
        
        # 5. 拼接所有片段的特征
        # [B, N, D] * num_segments -> [B, N_total, D]
        features = torch.cat(all_features, dim=1)
        
        # 验证输出
        if features.dim() != 3:
            raise ValueError(f"Expected 3D features [B, N, D], got shape: {features.shape}")
        
        return features
    
    def get_feature_info(self):
        """返回特征提取的信息（用于调试）"""
        # 计算 tokens 数量
        tokens_per_segment = (
            (self.native_frames // self.tubelet_size) * 
            (224 // self.patch_size) * 
            (224 // self.patch_size)
        )
        total_tokens = tokens_per_segment * self.num_segments
        
        return {
            'model_name': self.model_name,
            'embed_dim': self.embed_dim,
            'patch_size': self.patch_size,
            'tubelet_size': self.tubelet_size,
            'native_frames': self.native_frames,
            'target_frames': self.target_frames,
            'num_segments': self.num_segments,
            'tokens_per_segment': tokens_per_segment,
            'total_tokens': total_tokens,
        }


# 测试代码
if __name__ == "__main__":
    import numpy as np
    print("Testing VideoMAEAdapter with 64-frame Long Window")
    
    # 创建 adapter（64 帧长窗口）
    adapter = VideoMAEAdapter.from_config(
        resolution=224,
        frames_per_clip=64,  # 64 帧长窗口
        checkpoint=None,
        model_name='videomaev2_large'
    )
    
    # 冻结参数
    adapter.freeze()
    
    # 测试输入: [B=2, C=3, F=64, H=224, W=224]
    dummy_input = torch.randn(2, 3, 64, 224, 224)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Feature info: {adapter.get_feature_info()}")
    
    with torch.no_grad():
        output = adapter(dummy_input)
    
    print(f"\nOutput shape: {output.shape}")
    
    # 验证
    info = adapter.get_feature_info()
    expected_tokens = info['total_tokens']
    print(f"Expected shape: [2, {expected_tokens}, {adapter.embed_dim}]")
    
    assert output.shape == (2, expected_tokens, adapter.embed_dim), \
        f"Shape mismatch! Got {output.shape}, expected (2, {expected_tokens}, {adapter.embed_dim})"
    
    print("\n✓ VideoMAEAdapter 64-frame long window test passed!")
