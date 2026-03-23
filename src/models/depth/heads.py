import math
from typing import Iterable, List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_hw(value: Union[int, Sequence[int]]) -> Tuple[int, int]:
    if isinstance(value, Iterable):
        value = list(value)
        if len(value) == 1:
            return int(value[0]), int(value[0])
        if len(value) >= 2:
            return int(value[0]), int(value[1])
    return int(value), int(value)


def _needs_align(mode: str) -> bool:
    return mode in {"bilinear", "bicubic", "trilinear"}


class _ConvUpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, upsample_mode: str = "bilinear"):
        super().__init__()
        groups = min(32, out_ch)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=out_ch),
            nn.GELU(),
        )
        self.upsample_mode = upsample_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return F.interpolate(
            x,
            scale_factor=2.0,
            mode=self.upsample_mode,
            align_corners=False if _needs_align(self.upsample_mode) else None,
        )


class SurgicalDepthHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        decoder_channels: Sequence[int] = (512, 256, 128, 64),
        patch_size: int = 16,
        target_size: Union[int, Sequence[int]] = 256,
        num_feature_levels: int = 1,
        upsample_mode: str = "bilinear",
        activation: str = "softplus",
        min_depth: float = 0.0,
        max_depth: float | None = None,
    ):
        super().__init__()
        if len(decoder_channels) < 2:
            raise ValueError("decoder_channels must contain at least two entries.")
        target_h, target_w = _to_hw(target_size)
        if target_h % patch_size != 0 or target_w % patch_size != 0:
            raise ValueError(
                f"Target size {target_h}x{target_w} must be divisible by patch_size={patch_size}."
            )

        self.patch_size = patch_size
        self.target_size = (target_h, target_w)
        self.grid_size = (target_h // patch_size, target_w // patch_size)
        self.num_feature_levels = max(1, num_feature_levels)
        self.upsample_mode = upsample_mode
        self.min_depth = min_depth
        self.max_depth = max_depth

        proj_dim = decoder_channels[0]
        self.feature_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, proj_dim),
                )
                for _ in range(self.num_feature_levels)
            ]
        )

        fusion_in_dim = proj_dim * self.num_feature_levels
        self.fusion = nn.Conv2d(fusion_in_dim, proj_dim, kernel_size=1, bias=False)

        blocks: List[nn.Module] = []
        in_dim = proj_dim
        for out_dim in decoder_channels[1:]:
            blocks.append(_ConvUpBlock(in_dim, out_dim, upsample_mode=upsample_mode))
            in_dim = out_dim
        self.decoder = nn.Sequential(*blocks)

        head_dim = max(32, in_dim // 2)
        self.pred_head = nn.Sequential(
            nn.Conv2d(in_dim, head_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(head_dim, 1, kernel_size=1),
        )

        self.activation_type = activation
        self.use_sigmoid_scaling = False
        if activation == "softplus":
            self.activation = nn.Softplus()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
            self.use_sigmoid_scaling = True
        elif activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation is None or activation == "identity":
            self.activation = None
        else:
            raise ValueError(f"Unsupported activation '{activation}' for depth head.")

    def _tokens_to_spatial(self, tokens: torch.Tensor) -> torch.Tensor:
        B, N, C = tokens.shape
        expected_tokens = self.grid_size[0] * self.grid_size[1]
        if N != expected_tokens:
            side = int(math.sqrt(N))
            if side * side != N:
                raise ValueError(f"Cannot reshape {N} tokens into a square grid.")
            H, W = side, side
        else:
            H, W = self.grid_size
        x = tokens.transpose(1, 2).contiguous().view(B, C, H, W)
        if (H, W) != self.grid_size:
            x = F.interpolate(
                x,
                size=self.grid_size,
                mode=self.upsample_mode,
                align_corners=False if _needs_align(self.upsample_mode) else None,
            )
        return x

    def forward(self, features: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        if isinstance(features, torch.Tensor):
            features = [features]

        processed = []
        for idx in range(self.num_feature_levels):
            feat = features[min(idx, len(features) - 1)]
            proj = self.feature_projections[idx](feat)
            spatial = self._tokens_to_spatial(proj)
            processed.append(spatial)

        fused = torch.cat(processed, dim=1)
        fused = self.fusion(fused)

        x = fused
        x = self.decoder(x)
        x = self.pred_head(x)
        x = F.interpolate(
            x,
            size=self.target_size,
            mode=self.upsample_mode,
            align_corners=False if _needs_align(self.upsample_mode) else None,
        )

        if self.activation is not None:
            x = self.activation(x)
        if self.use_sigmoid_scaling and self.max_depth is not None:
            span = self.max_depth - self.min_depth
            x = x * span + self.min_depth
        else:
            if self.min_depth:
                x = x + self.min_depth
            if self.max_depth is not None:
                x = torch.clamp(x, max=self.max_depth)
        return x
