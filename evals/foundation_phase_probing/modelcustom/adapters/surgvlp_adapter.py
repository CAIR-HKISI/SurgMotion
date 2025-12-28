import torch
import torch.nn as nn
import sys
import os
from typing import Optional
import importlib
import importlib.util

# Add foundation_models/SurgVLP to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming repository root is 4 levels up from this file
# evals/foundation_phase_probing/modelcustom/adapters/surgvlp_adapter.py -> root
repo_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
surgvlp_path = os.path.join(repo_root, "foundation_models/SurgVLP")

if surgvlp_path not in sys.path:
    # 插到最前面，避免被环境里“同名 surgvlp 包”抢占
    sys.path.insert(0, surgvlp_path)

# 关键：检查 sys.modules 缓存。如果之前已经导入了错误的 surgvlp（例如环境自带的），
# 即使修改了 sys.path，Python 也会直接用缓存。必须强制清理。
if "surgvlp" in sys.modules:
    _loaded = sys.modules["surgvlp"]
    # 如果已加载的模块没有 load 方法，或者路径不对，就删掉缓存重来
    if not hasattr(_loaded, "load"):
        print(f"Warning: 检测到 sys.modules 中已存在不兼容的 surgvlp ({getattr(_loaded, '__file__', 'unknown')})，正在移除以重新加载...")
        del sys.modules["surgvlp"]
        # 也要清理可能的子模块缓存，防止部分残留
        for k in list(sys.modules.keys()):
            if k.startswith("surgvlp."):
                del sys.modules[k]

try:
    import surgvlp
    # 再次检查，确保是我们要的那个
    if not hasattr(surgvlp, "load"):
        # 可能是导入了 surgvlp 包，但 __init__ 没暴露 load？尝试导入子模块
        import surgvlp.surgvlp
        if hasattr(surgvlp.surgvlp, "load"):
             surgvlp = surgvlp.surgvlp
    
    _SURGVLP_IMPORT_ERROR: Optional[BaseException] = None
except Exception as e:
    surgvlp = None  # type: ignore[assignment]
    _SURGVLP_IMPORT_ERROR = e
    print(f"Warning: Could not import surgvlp from {surgvlp_path}: {type(e).__name__}: {e}")

from .base_adapter import BaseFoundationModelAdapter

class SurgVLPAdapter(BaseFoundationModelAdapter):
    """SurgVLP Adapter - Input format: [B, C, F, H, W]"""
    
    def __init__(self, model, embed_dim: int, model_name: str, resolution: int = 224):
        super().__init__(model, embed_dim)
        self.model_name = model_name
        self.resolution = resolution
        
    @classmethod
    def from_config(cls, resolution: int = 224, checkpoint: Optional[str] = None, model_name: str = 'SurgVLP'):
        """Create adapter from config"""
        
        if surgvlp is None:
            # 针对性检查关键文件，不再打印冗长的目录列表
            missing_files = []
            target_pkg_dir = os.path.join(surgvlp_path, "surgvlp")
            
            if not os.path.exists(os.path.join(target_pkg_dir, "codes")):
                missing_files.append("surgvlp/codes/ (目录缺失)")
            if not os.path.exists(os.path.join(target_pkg_dir, "__init__.py")):
                missing_files.append("surgvlp/__init__.py (文件缺失)")
            
            hint = ""
            if missing_files:
                hint = f"\n** 检测到文件缺失: {', '.join(missing_files)} **\n这通常是由于文件同步/上传不完整导致的。"

            raise RuntimeError(
                "无法导入 surgvlp（SurgVLP foundation model 代码）。\n"
                f"- 期望路径: {surgvlp_path}\n"
                f"- 原始异常: {type(_SURGVLP_IMPORT_ERROR).__name__}: {_SURGVLP_IMPORT_ERROR}{hint}\n"
                "请确认：\n"
                "1) 务必重新同步 foundation_models/SurgVLP 文件夹（确保包含 codes 子目录）。\n"
                "2) 依赖已安装（至少 torch）。\n"
                "3) 或者先在 SurgVLP 目录执行 `pip install -e .`。\n"
            )

        # Define config directly to avoid mmengine config parsing issues with transforms
        model_config = dict(
            type='SurgVLP',
            backbone_img = dict(
                type='img_backbones/ImageEncoder',
                num_classes=768,
                pretrained='imagenet',
                backbone_name='resnet_50',
                img_norm=False
            ),
            # backbone_text removed to avoid dependency on transformers
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if checkpoint is None:
            checkpoint = "ckpts/ckpts_foundation/SurgVLP.pth"
        
        print(f"Loading SurgVLP model from {checkpoint}...")
        # Load model using surgvlp.load
        # pretrain arg can be used to load specific checkpoint
        model, _ = surgvlp.load(model_config, device=device, pretrain=checkpoint)
        
        # Determine embed_dim
        # ResNet50 layer4 output channels is 2048
        embed_dim = 2048
        
        print(f"✓ SurgVLP loaded: embed_dim={embed_dim}")
        
        return cls(model, embed_dim, model_name, resolution)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, F, H, W] Video input
        Returns:
            features: [B, F*N, D]
        """
        B, C, F, H, W = x.shape
        
        target_H = self.resolution
        target_W = self.resolution
        
        # Flatten time dimension for processing
        # [B, C, F, H, W] -> [B*F, C, H, W]
        x = x.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
        
        # Resize if needed
        if H != target_H or W != target_W:
            import torch.nn.functional as Fn
            x = Fn.interpolate(
                x, 
                size=(target_H, target_W), 
                mode='bilinear', 
                align_corners=False
            )
            
        # Extract features using internal ResNet backbone
        # model -> backbone_img (ImageEncoder) -> model (ResNet)
        if hasattr(self.model, 'backbone_img') and hasattr(self.model.backbone_img, 'model'):
            resnet = self.model.backbone_img.model
            
            # Forward pass through ResNet up to layer4
            x = resnet.conv1(x)
            x = resnet.bn1(x)
            x = resnet.relu(x)
            x = resnet.maxpool(x)

            x = resnet.layer1(x)
            x = resnet.layer2(x)
            x = resnet.layer3(x)
            x = resnet.layer4(x)
            
            # Output shape: [B*F, 2048, H/32, W/32]
            
            # Flatten spatial dimensions: [B*F, C, H', W'] -> [B*F, C, N] -> [B*F, N, C]
            x = x.flatten(2).transpose(1, 2)
            
        else:
            # Fallback if structure is different (should not happen with standard SurgVLP)
            raise AttributeError("Could not find ResNet backbone in SurgVLP model")
            
        # Reshape back to include time dimension
        # [B*F, N, D] -> [B, F, N, D] -> [B, F*N, D]
        BF, N, D = x.shape
        x = x.reshape(B, F, N, D).reshape(B, F * N, D)
        
        return x

    def get_feature_info(self):
        return {
            'model_name': self.model_name,
            'embed_dim': self.embed_dim,
            'resolution': self.resolution
        }

if __name__ == "__main__":
    # Test code
    try:
        adapter = SurgVLPAdapter.from_config(resolution=224)
        dummy_input = torch.randn(2, 3, 4, 224, 224).cuda()
        with torch.no_grad():
            output = adapter(dummy_input)
        print(f"Input: {dummy_input.shape}")
        print(f"Output: {output.shape}")
    except Exception as e:
        print(f"Test failed: {e}")
