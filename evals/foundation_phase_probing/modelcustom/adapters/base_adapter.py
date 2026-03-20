import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class BaseFoundationModelAdapter(nn.Module):
    """
    Base adapter class for Foundation Models.
    All concrete adapters should inherit from this class.
    """
    
    def __init__(self, model, embed_dim: int):
        super().__init__()
        self.model = model
        self.embed_dim = embed_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, F, H, W] or [B*num_clips, C, F, H, W]
        Returns:
            features: [B, N, D] where N is number of tokens, D is embed_dim
        """
        raise NotImplementedError("Adapter must implement forward method")
    
    def freeze(self):
        """Freeze model parameters (for probing)."""
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info(f"✓ Frozen {self.__class__.__name__}")