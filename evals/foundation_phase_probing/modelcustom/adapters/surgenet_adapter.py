"""
SurgeNet Foundation Model Adapter

基于 SurgeNet 仓库 (https://github.com/TimJaspers0801/SurgeNet)
支持 CAFormer-S18 backbone 及其变体 (ConvNextv2, PVTv2)

SurgeNet 是在大规模手术视频数据上使用 DINO 自监督学习预训练的模型，
在手术场景分割、阶段识别等任务上表现优异。
"""

import sys
import torch
import torch.nn as nn
from typing import Optional
from pathlib import Path
from .base_adapter import BaseFoundationModelAdapter

# Ensure current directory is in path for relative imports if needed
if "." not in sys.path:
    sys.path.append(".")

# SurgeNet 预训练权重 URL
SURGENET_URLS = {
    "surgenetxl": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/SurgeNetXL_checkpoint_epoch0050_teacher.pth?download=true",
    "surgenet": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/SurgeNet_checkpoint_epoch0050_teacher.pth?download=true",
    "surgenet_small": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/SurgeNetSmall_checkpoint_epoch0050_teacher.pth?download=true",
    "surgenet_cholec": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/CHOLEC_checkpoint_epoch0050_teacher.pth?download=true",
    "surgenet_ramie": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/RAMIE_checkpoint_epoch0050_teacher.pth?download=true",
    "surgenet_rarp": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/RARP_checkpoint_epoch0050_teacher.pth?download=true",
    "surgenet_public": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/Public_checkpoint0050.pth?download=true",
    "surgenet_convnextv2": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/SurgeNet_ConvNextv2_checkpoint_epoch0050_teacher.pth?download=true",
    "surgenet_pvtv2": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/SurgeNet_PVTv2_checkpoint_epoch0050_teacher.pth?download=true",
}

# 本地权重文件名映射
SURGENET_FILENAMES = {
    "surgenetxl": "SurgeNetXL_checkpoint_epoch0050_teacher.pth",
    "surgenet_convnextv2": "SurgeNet_ConvNextv2_checkpoint_epoch0050_teacher.pth",
}

# 本地权重目录
LOCAL_CHECKPOINT_DIR = Path("ckpts/ckpts_foundation/SurgeNetModels")

# 不同 backbone 的输出维度
BACKBONE_DIMS = {
    "caformer": 512,       # CAFormer-S18
    "convnextv2": 768,     # ConvNextv2-Tiny
    "pvtv2": 512,          # PVTv2-B2
}


class SurgeNetAdapter(BaseFoundationModelAdapter):
    """
    SurgeNet 模型的 Adapter - 输入格式: [B, C, F, H, W]
    
    SurgeNet 使用 CAFormer-S18 (MetaFormer架构) 作为默认 backbone，
    也支持 ConvNextv2 和 PVTv2 作为备选 backbone。
    
    模型输出维度:
    - CAFormer-S18: 512
    - ConvNextv2-Tiny: 768
    - PVTv2-B2: 512
    """
    
    def __init__(self, model, embed_dim: int, model_name: str, backbone_type: str = "caformer"):
        super().__init__(model, embed_dim)
        self.model_name = model_name
        self.backbone_type = backbone_type
        self._first_forward = True  # 用于控制日志输出
    
    @classmethod
    def from_config(
        cls,
        resolution: int,
        checkpoint: Optional[str] = None,
        model_name: str = 'surgenetxl',
        backbone_type: Optional[str] = None
    ):
        """
        从配置创建 SurgeNet adapter
        
        Args:
            resolution: 输入分辨率 (建议使用 224)
            checkpoint: 本地 checkpoint 路径 (如果为 None，从 HuggingFace 下载)
            model_name: 模型名称
            backbone_type: backbone类型 ('caformer', 'convnextv2', 'pvtv2')。
                          如果为 None，将尝试从 model_name 推断。
        
        Returns:
            SurgeNetAdapter 实例
        """
        # 1. 推断 backbone_type
        if backbone_type is None:
            if 'convnextv2' in model_name.lower():
                backbone_type = 'convnextv2'
            elif 'pvtv2' in model_name.lower():
                backbone_type = 'pvtv2'
            else:
                backbone_type = 'caformer'
                
        print(f"Loading SurgeNet model: {model_name} (backbone: {backbone_type})")
        
        # 2. 尝试自动查找本地权重
        if checkpoint is None:
            model_key = model_name.lower().replace('-', '_')
            if model_key in SURGENET_FILENAMES:
                local_path = LOCAL_CHECKPOINT_DIR / SURGENET_FILENAMES[model_key]
                if local_path.exists():
                    checkpoint = str(local_path)
                    print(f"Found local checkpoint: {checkpoint}")
        
        # 3. 添加 SurgeNet 仓库路径到 sys.path
        surgenet_path = Path(__file__).parent.parent.parent.parent.parent / "foundation_models" / "SurgeNet"
        if str(surgenet_path) not in sys.path:
            sys.path.insert(0, str(surgenet_path))
            print(f"Added SurgeNet path: {surgenet_path}")
        
        try:
            # 4. 根据 backbone 类型加载模型
            if backbone_type == 'caformer':
                from foundation_models.SurgeNet.metaformer import caformer_s18
                
                # 确定权重 URL
                model_key = model_name.lower().replace('-', '_')
                weights_url = SURGENET_URLS.get(model_key, SURGENET_URLS.get('surgenetxl'))
                
                # 如果有本地 checkpoint，不通过 URL 加载
                if checkpoint and Path(checkpoint).exists():
                    weights_url = None
                    print(f"Using local checkpoint: {checkpoint}")
                
                # 创建模型
                model = caformer_s18(
                    num_classes=0,
                    pretrained='SurgeNet',
                    pretrained_weights=weights_url
                )
                
                embed_dim = BACKBONE_DIMS['caformer']
                
            elif backbone_type == 'convnextv2':
                from foundation_models.SurgeNet.convnextv2 import convnextv2_tiny
                
                weights_url = SURGENET_URLS.get('surgenet_convnextv2')
                
                # 如果有本地 checkpoint，不通过 URL 加载
                if checkpoint and Path(checkpoint).exists():
                    weights_url = None
                
                # 创建模型 (pretrained_weights=None 避免自动下载)
                model = convnextv2_tiny(
                    num_classes=0,
                    pretrained_weights=None if checkpoint else weights_url
                )
                
                embed_dim = BACKBONE_DIMS['convnextv2']
                
            elif backbone_type == 'pvtv2':
                from foundation_models.SurgeNet.pvtv2 import pvt_v2_b2
                
                weights_url = SURGENET_URLS.get('surgenet_pvtv2')
                if checkpoint and Path(checkpoint).exists():
                    weights_url = checkpoint # pvt_v2 implementation handles path string as local path too usually
                
                model = pvt_v2_b2(
                    num_classes=0,
                    pretrained_weights=weights_url
                )
                embed_dim = BACKBONE_DIMS['pvtv2']
            
            else:
                raise ValueError(f"Unsupported backbone type: {backbone_type}")
            
            # 5. 加载本地权重 (通用逻辑)
            if checkpoint and Path(checkpoint).exists():
                state_dict = torch.load(checkpoint, map_location='cpu')
                
                # 处理可能的嵌套结构
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif 'model' in state_dict:
                    state_dict = state_dict['model']
                
                # 移除分类头权重
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head.')}
                
                # 移除可能的前缀 (针对 CAFormer 等)
                for prefix in ['module.', 'backbone.', 'encoder.']:
                    state_dict = {k.replace(prefix, ''): v for k, v in state_dict.items()}
                
                msg = model.load_state_dict(state_dict, strict=False)
                print(f"Loaded checkpoint: {msg}")
            
            print(f"✓ SurgeNet loaded successfully:")
            print(f"  - Model: {model_name}")
            print(f"  - Backbone: {backbone_type}")
            print(f"  - embed_dim: {embed_dim}")
            
        except Exception as e:
            print(f"Error loading SurgeNet model: {e}")
            import traceback
            traceback.print_exc()
            raise e
        
        return cls(model, embed_dim, model_name, backbone_type)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, F, H, W] 视频输入
        Returns:
            features: [B, F*N, D] 
        """
        B, C, F, H, W = x.shape
        
        # 标准输入尺寸 (SurgeNet 推荐使用 224x224)
        target_H, target_W = 224, 224
        
        # 如果尺寸不匹配，进行 resize
        if H != target_H or W != target_W:
            import torch.nn.functional as fn
            x = x.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
            x = fn.interpolate(x, size=(target_H, target_W), mode='bilinear', align_corners=False)
            x = x.reshape(B, F, C, target_H, target_W).permute(0, 2, 1, 3, 4)
            H, W = target_H, target_W
        
        # SurgeNet 是图像模型，需要将时间维度展开
        x = x.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
        
        # 通过 SurgeNet 提取特征
        with torch.no_grad():
            if self.backbone_type == 'convnextv2':
                 features = self.model.forward_features(x)
                 # Handle varied return types (tensor, tuple, list)
                 if isinstance(features, (tuple, list)):
                     feature_map = features[-1] if isinstance(features[-1], list) else features[-1]
                     if isinstance(feature_map, (tuple, list)): # Nested list check
                         feature_map = feature_map[-1]
                 else:
                     feature_map = features
            else:
                # CAFormer returns (global_features, feature_list)
                outputs = self.model.forward_features(x)
                if isinstance(outputs, tuple) and len(outputs) == 2:
                    global_features, feature_list = outputs
                    feature_map = feature_list[-1]
                else:
                    feature_map = outputs

        # feature_map shape: [B*F, C', H', W']
        BF, C_out, H_feat, W_feat = feature_map.shape
        
        # Flatten spatial dimensions: [B*F, C', H', W'] -> [B*F, C', N] -> [B*F, N, C']
        features = feature_map.flatten(2).permute(0, 2, 1)
        
        # Restructure to [B, F*N, D]
        N = H_feat * W_feat
        features = features.reshape(B, F * N, C_out)
        
        if self._first_forward:
            print(f"📊 SurgeNet feature extraction:")
            print(f"   Input: [{B}, {C}, {F}, {H}, {W}]")
            print(f"   Output: {features.shape}")
            self._first_forward = False
        
        return features
    
    def get_feature_info(self):
        return {
            'model_name': self.model_name,
            'embed_dim': self.embed_dim,
            'backbone': self.backbone_type,
            'pretrain': 'DINO (SurgeNet)',
        }


if __name__ == "__main__":
    print("Testing SurgeNet Adapter")
    
    # 1. Test SurgeNetXL (CAFormer)
    print("\n1. Testing SurgeNetXL (CAFormer)...")
    try:
        adapter = SurgeNetAdapter.from_config(resolution=224, model_name='surgenetxl')
        dummy_input = torch.randn(1, 3, 4, 224, 224)
        output = adapter(dummy_input)
        print(f"  ✓ Output shape: {output.shape} (Expected: [1, 196, 512])")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()

    # 2. Test SurgeNet ConvNextv2
    print("\n2. Testing SurgeNet ConvNextv2...")
    try:
        adapter = SurgeNetAdapter.from_config(resolution=224, model_name='surgenet_convnextv2')
        dummy_input = torch.randn(1, 3, 4, 224, 224)
        output = adapter(dummy_input)
        print(f"  ✓ Output shape: {output.shape} (Expected: [1, 196, 768])")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
