import logging
import torch
import torch.nn as nn

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class BaseFoundationModelAdapter(nn.Module):
    """
    Foundation Model的基础Adapter类
    所有具体的adapter都应该继承这个类
    """
    
    def __init__(self, model, embed_dim: int):
        super().__init__()
        self.model = model
        self.embed_dim = embed_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, F, H, W] 或 [B*num_clips, C, F, H, W]
        Returns:
            features: [B, N, D] where N is number of tokens, D is embed_dim
        """
        raise NotImplementedError("Adapter must implement forward method")
    
    def freeze(self):
        """冻结模型参数（用于probing）"""
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info(f"✓ Frozen {self.__class__.__name__}")