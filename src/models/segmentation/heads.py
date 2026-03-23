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


class _MultiheadAttention(nn.Module):
    """
    Thin wrapper over torch.nn.MultiheadAttention supporting both batch_first and legacy shapes.
    Expects inputs/outputs as [B, L, C].
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.batch_first = True
        self.num_heads = int(num_heads)
        try:
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        except TypeError:
            self.batch_first = False
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.batch_first:
            out, _ = self.attn(query, key, value, need_weights=False, attn_mask=attn_mask)
            return out
        q = query.transpose(0, 1)
        k = key.transpose(0, 1)
        v = value.transpose(0, 1)
        out, _ = self.attn(q, k, v, need_weights=False, attn_mask=attn_mask)
        return out.transpose(0, 1)


class _DecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        nheads: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.self_attn = _MultiheadAttention(d_model, nheads, dropout=dropout)
        self.cross_attn = _MultiheadAttention(d_model, nheads, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.act = nn.GELU()

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, *, cross_attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = tgt
        x2 = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(x2))
        x2 = self.cross_attn(x, memory, memory, attn_mask=cross_attn_mask)
        x = self.norm2(x + self.dropout(x2))
        x2 = self.linear2(self.dropout(self.act(self.linear1(x))))
        x = self.norm3(x + self.dropout(x2))
        return x


class MaskFormerSegmentationHead(nn.Module):
    """
    MaskFormer-like segmentation head for *binary semantic segmentation*.

    Frozen ViT produces token features (optionally multiple layers). We:
    - project ViT tokens -> dense mask features at image resolution (pixel decoder)
    - decode a small set of learnable queries with cross-attn to token memory
    - turn queries into mask embeddings and dot-product with mask features to get mask logits
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        patch_size: int = 16,
        target_size: Union[int, Sequence[int]] = 384,
        num_feature_levels: int = 1,
        hidden_dim: int = 256,
        mask_dim: int = 256,
        num_queries: int = 1,
        num_decoder_layers: int = 4,
        nheads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        decoder_channels: Sequence[int] = (256, 128, 64),
        upsample_mode: str = "bilinear",
        return_aux: bool = True,
    ):
        super().__init__()
        target_h, target_w = _to_hw(target_size)
        if target_h % patch_size != 0 or target_w % patch_size != 0:
            raise ValueError(f"target_size {target_h}x{target_w} must be divisible by patch_size={patch_size}.")

        self.patch_size = int(patch_size)
        self.target_size = (target_h, target_w)
        self.grid_size = (target_h // patch_size, target_w // patch_size)
        self.num_feature_levels = max(1, int(num_feature_levels))
        self.hidden_dim = int(hidden_dim)
        self.mask_dim = int(mask_dim)
        self.num_queries = max(1, int(num_queries))
        self.return_aux = bool(return_aux)
        self.upsample_mode = str(upsample_mode)

        self.feature_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, self.hidden_dim),
                )
                for _ in range(self.num_feature_levels)
            ]
        )

        self.pixel_fusion = nn.Conv2d(self.hidden_dim * self.num_feature_levels, self.hidden_dim, kernel_size=1, bias=False)

        channels = list(decoder_channels)
        if len(channels) == 0:
            channels = [self.hidden_dim]
        if channels[0] != self.hidden_dim:
            self.decoder_input_proj = nn.Conv2d(self.hidden_dim, channels[0], kernel_size=1, bias=False)
        else:
            self.decoder_input_proj = nn.Identity()

        blocks: List[nn.Module] = []
        for in_c, out_c in zip(channels, channels[1:]):
            blocks.append(_ConvUpBlock(in_c, out_c, upsample_mode=self.upsample_mode))
        self.pixel_decoder = nn.Sequential(*blocks)
        final_c = channels[-1]
        groups = min(32, self.mask_dim)
        self.mask_feature_proj = nn.Sequential(
            nn.Conv2d(final_c, self.mask_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=self.mask_dim),
            nn.GELU(),
        )

        self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)
        self.decoder_layers = nn.ModuleList(
            [
                _DecoderLayer(
                    d_model=self.hidden_dim,
                    nheads=int(nheads),
                    dim_feedforward=int(dim_feedforward),
                    dropout=float(dropout),
                )
                for _ in range(int(num_decoder_layers))
            ]
        )

        self.mask_embed = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.mask_dim),
        )

    def _tokens_to_spatial(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, N, C] -> [B, C, H_grid, W_grid]
        """
        B, N, C = tokens.shape
        expected = self.grid_size[0] * self.grid_size[1]
        if N != expected:
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

    def forward(self, features: Union[torch.Tensor, List[torch.Tensor]]) -> dict:
        """
        features:
          - Tensor [B, N, embed_dim] OR
          - List[Tensor] of length >= 1, each [B, N, embed_dim]
        Returns:
          {"logits": [B,1,H,W], "aux_logits": List[[B,1,H,W]]}
        """
        if isinstance(features, torch.Tensor):
            features = [features]

        projected_tokens: List[torch.Tensor] = []
        spatial_feats: List[torch.Tensor] = []
        for idx in range(self.num_feature_levels):
            feat = features[min(idx, len(features) - 1)]
            proj = self.feature_projections[idx](feat)  # [B,N,hidden]
            projected_tokens.append(proj)
            spatial_feats.append(self._tokens_to_spatial(proj))

        memory = torch.stack(projected_tokens, dim=0).mean(dim=0)  # [B,N,hidden]

        fused = torch.cat(spatial_feats, dim=1)
        fused = self.pixel_fusion(fused)
        x = self.decoder_input_proj(fused)
        x = self.pixel_decoder(x)
        mask_features = self.mask_feature_proj(x)
        mask_features = F.interpolate(
            mask_features,
            size=self.target_size,
            mode=self.upsample_mode,
            align_corners=False if _needs_align(self.upsample_mode) else None,
        )

        B = memory.shape[0]
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B,Q,hidden]

        aux_logits: List[torch.Tensor] = []
        q = queries
        for layer_idx, layer in enumerate(self.decoder_layers):
            q = layer(q, memory)
            if self.return_aux and layer_idx < len(self.decoder_layers) - 1:
                aux_logits.append(self._predict_masks(q, mask_features))

        logits = self._predict_masks(q, mask_features)
        return {"logits": logits, "aux_logits": aux_logits}

    def _predict_masks(self, q: torch.Tensor, mask_features: torch.Tensor) -> torch.Tensor:
        """
        q: [B,Q,hidden], mask_features: [B,mask_dim,H,W] -> logits [B,1,H,W]
        """
        mask_embed = self.mask_embed(q)  # [B,Q,mask_dim]
        mask_logits = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        # Binary semantic segmentation: use first query as the foreground mask.
        return mask_logits[:, 0:1, :, :]


class Mask2FormerSegmentationHead(MaskFormerSegmentationHead):
    """
    Minimal Mask2Former-style variant: identical to MaskFormer head, but uses
    *masked cross-attention* in the transformer decoder.

    At each decoder layer we:
    - predict a mask from the current queries
    - convert it to a boolean attention mask over patch tokens
    - use that mask to restrict cross-attention in the next layer
    """

    def __init__(
        self,
        *,
        attn_mask_threshold: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.attn_mask_threshold = float(attn_mask_threshold)

        # Best-effort: infer #heads from the first decoder layer.
        try:
            self._nheads = int(self.decoder_layers[0].cross_attn.num_heads)
        except Exception:
            self._nheads = 1

    def forward(self, features: Union[torch.Tensor, List[torch.Tensor]]) -> dict:
        if isinstance(features, torch.Tensor):
            features = [features]

        projected_tokens: List[torch.Tensor] = []
        spatial_feats: List[torch.Tensor] = []
        for idx in range(self.num_feature_levels):
            feat = features[min(idx, len(features) - 1)]
            proj = self.feature_projections[idx](feat)  # [B,N,hidden]
            projected_tokens.append(proj)
            spatial_feats.append(self._tokens_to_spatial(proj))

        memory = torch.stack(projected_tokens, dim=0).mean(dim=0)  # [B,N,hidden]

        fused = torch.cat(spatial_feats, dim=1)
        fused = self.pixel_fusion(fused)
        x = self.decoder_input_proj(fused)
        x = self.pixel_decoder(x)
        mask_features = self.mask_feature_proj(x)
        mask_features = F.interpolate(
            mask_features,
            size=self.target_size,
            mode=self.upsample_mode,
            align_corners=False if _needs_align(self.upsample_mode) else None,
        )

        B = memory.shape[0]
        q = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B,Q,hidden]

        aux_logits: List[torch.Tensor] = []
        cross_attn_mask = None
        last_logits = None
        for layer_idx, layer in enumerate(self.decoder_layers):
            q = layer(q, memory, cross_attn_mask=cross_attn_mask)
            last_logits = self._predict_masks(q, mask_features)
            if self.return_aux and layer_idx < len(self.decoder_layers) - 1:
                aux_logits.append(last_logits)
            cross_attn_mask = self._build_cross_attn_mask(last_logits, num_heads=self._nheads)

        if last_logits is None:
            last_logits = self._predict_masks(q, mask_features)
        return {"logits": last_logits, "aux_logits": aux_logits}

    def _build_cross_attn_mask(self, mask_logits: torch.Tensor, *, num_heads: int) -> torch.Tensor | None:
        """
        mask_logits: [B,1,H,W] -> attn_mask: [B*num_heads, Q, N_tokens] (bool)
        True = masked (not allowed).
        """
        # We mask patch tokens based on predicted mask on the patch grid.
        with torch.no_grad():
            probs = torch.sigmoid(mask_logits.detach())  # [B,1,H,W]
            probs_grid = F.interpolate(
                probs,
                size=self.grid_size,
                mode="bilinear",
                align_corners=False,
            )
            probs_flat = probs_grid.flatten(2)  # [B,1,N]
            allow = probs_flat >= self.attn_mask_threshold
            # If a mask is empty, do not mask anything for that sample.
            empty = ~allow.any(dim=2, keepdim=True)  # [B,1,1]
            allow = torch.where(empty, torch.ones_like(allow, dtype=torch.bool), allow)
            deny = ~allow  # [B,1,N] boolean

            # Expand to [B, Q, N]. (Binary head uses Q=1, but keep general.)
            B, Q, N = deny.shape[0], self.num_queries, deny.shape[-1]
            deny_q = deny.expand(B, Q, N)

            # MultiheadAttention expects (B*num_heads, Q, N) for per-sample masks.
            if num_heads <= 1:
                return deny_q
            mask = deny_q.unsqueeze(1).expand(B, num_heads, Q, N).reshape(B * num_heads, Q, N)
            return mask


class Mask2FormerSegmentationHeadPaper(MaskFormerSegmentationHead):
    """
    Closer-to-paper Mask2Former-style variant.

    Key differences vs our lightweight `Mask2FormerSegmentationHead`:
    - Cross-attention is performed over *multi-scale pixel features* (flattened HxW features), not ViT token memory.
    - Masked attention mask is built from predicted masks and resized to the *next* feature level resolution,
      then applied to cross-attention in the next decoder layer (round-robin over feature levels).

    This is still not a full Detectron2 Mask2Former reproduction (no deformable-attn pixel decoder, no Hungarian matching,
    no class logits / mask classification, no point-sampling loss), but it matches the core masked cross-attn mechanism
    and multi-scale cycling behavior from the paper.
    """

    def __init__(
        self,
        *,
        attn_mask_threshold: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.attn_mask_threshold = float(attn_mask_threshold)

        # Build per-pixel-level projections to hidden_dim for attention memory.
        # Levels are: fused (hidden_dim at patch-grid) + outputs after each upsample block (decoder_channels[1:]).
        decoder_channels = list(kwargs.get("decoder_channels", (self.hidden_dim,)))
        if len(decoder_channels) == 0:
            decoder_channels = [self.hidden_dim]
        self._pixel_level_channels = [self.hidden_dim] + decoder_channels[1:]
        projs: List[nn.Module] = [nn.Identity()]
        for ch in self._pixel_level_channels[1:]:
            if int(ch) == int(self.hidden_dim):
                projs.append(nn.Identity())
            else:
                projs.append(nn.Conv2d(int(ch), int(self.hidden_dim), kernel_size=1, bias=False))
        self.pixel_level_projs = nn.ModuleList(projs)

        # Best-effort: infer #heads from the first decoder layer.
        try:
            self._nheads = int(self.decoder_layers[0].cross_attn.num_heads)
        except Exception:
            self._nheads = 1

    def forward(self, features: Union[torch.Tensor, List[torch.Tensor]]) -> dict:
        if isinstance(features, torch.Tensor):
            features = [features]

        spatial_feats: List[torch.Tensor] = []
        for idx in range(self.num_feature_levels):
            feat = features[min(idx, len(features) - 1)]
            proj = self.feature_projections[idx](feat)  # [B,N,hidden]
            spatial_feats.append(self._tokens_to_spatial(proj))  # [B,hidden,Hg,Wg]

        fused = torch.cat(spatial_feats, dim=1)
        fused = self.pixel_fusion(fused)  # [B,hidden,Hg,Wg]

        # Build multi-scale pixel features for attention memory.
        pixel_levels_raw: List[torch.Tensor] = [fused]
        x = self.decoder_input_proj(fused)
        for blk in self.pixel_decoder:
            x = blk(x)
            pixel_levels_raw.append(x)

        pixel_levels: List[torch.Tensor] = []
        level_sizes: List[Tuple[int, int]] = []
        for lvl, feat in enumerate(pixel_levels_raw):
            feat = self.pixel_level_projs[min(lvl, len(self.pixel_level_projs) - 1)](feat)
            pixel_levels.append(feat)
            level_sizes.append((int(feat.shape[-2]), int(feat.shape[-1])))

        # Mask features for mask prediction.
        mask_features = self.mask_feature_proj(x)
        mask_features = F.interpolate(
            mask_features,
            size=self.target_size,
            mode=self.upsample_mode,
            align_corners=False if _needs_align(self.upsample_mode) else None,
        )

        B = fused.shape[0]
        Q = self.num_queries
        q = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B,Q,hidden]

        aux_logits: List[torch.Tensor] = []
        cross_attn_mask = None
        last_mask_logits_all = None

        num_levels = max(1, len(pixel_levels))
        for layer_idx, layer in enumerate(self.decoder_layers):
            lvl = layer_idx % num_levels
            mem_feat = pixel_levels[lvl]  # [B,hidden,H,W]
            mem = mem_feat.flatten(2).transpose(1, 2).contiguous()  # [B,HW,hidden]

            q = layer(q, mem, cross_attn_mask=cross_attn_mask)

            # Predict masks for all queries (for attention masking); return first query as binary output.
            mask_embed = self.mask_embed(q)  # [B,Q,mask_dim]
            mask_logits_all = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)  # [B,Q,H,W]
            last_mask_logits_all = mask_logits_all

            out_logits = mask_logits_all[:, 0:1, :, :]
            if self.return_aux and layer_idx < len(self.decoder_layers) - 1:
                aux_logits.append(out_logits)

            # Build mask for the *next* feature level (round-robin), like the paper implementation.
            next_lvl = (layer_idx + 1) % num_levels
            cross_attn_mask = self._build_cross_attn_mask_from_logits(
                mask_logits_all,
                target_hw=level_sizes[next_lvl],
                num_heads=self._nheads,
            )

        if last_mask_logits_all is None:
            mask_embed = self.mask_embed(q)
            last_mask_logits_all = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        logits = last_mask_logits_all[:, 0:1, :, :]
        return {"logits": logits, "aux_logits": aux_logits}

    def _build_cross_attn_mask_from_logits(
        self,
        mask_logits: torch.Tensor,
        *,
        target_hw: Tuple[int, int],
        num_heads: int,
    ) -> torch.Tensor:
        """
        mask_logits: [B,Q,H,W] -> attn_mask: [B*num_heads, Q, HW] (bool)
        True = masked (not allowed).
        """
        with torch.no_grad():
            probs = torch.sigmoid(mask_logits.detach())
            probs = F.interpolate(
                probs,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
            probs_flat = probs.flatten(2)  # [B,Q,N]
            allow = probs_flat >= self.attn_mask_threshold
            # If a query mask is empty, do not mask anything for that (B,Q).
            empty = ~allow.any(dim=2, keepdim=True)  # [B,Q,1]
            allow = allow | empty  # broadcast along N
            deny = ~allow  # [B,Q,N]

            if num_heads <= 1:
                return deny
            B, Q, N = deny.shape
            return deny.unsqueeze(1).expand(B, num_heads, Q, N).reshape(B * num_heads, Q, N)
