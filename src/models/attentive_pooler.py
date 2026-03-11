# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import math

import torch
import torch.nn as nn

from src.models.utils.modules import Block, CrossAttention, CrossAttentionBlock
from src.utils.tensors import trunc_normal_


class AttentivePooler(nn.Module):
    """Attentive Pooler"""

    def __init__(
        self,
        num_queries=1,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        complete_block=True,
        use_activation_checkpointing=False,
    ):
        super().__init__()
        self.use_activation_checkpointing = use_activation_checkpointing
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, embed_dim))

        self.complete_block = complete_block
        if complete_block:
            self.cross_attention_block = CrossAttentionBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer
            )
        else:
            self.cross_attention_block = CrossAttention(dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias)

        self.blocks = None
        if depth > 1:
            self.blocks = nn.ModuleList(
                [
                    Block(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=False,
                        norm_layer=norm_layer,
                    )
                    for i in range(depth - 1)
                ]
            )

        self.init_std = init_std
        trunc_normal_(self.query_tokens, std=self.init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        layer_id = 0
        if self.blocks is not None:
            for layer_id, layer in enumerate(self.blocks):
                rescale(layer.attn.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

        if self.complete_block:
            rescale(self.cross_attention_block.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.blocks is not None:
            for blk in self.blocks:
                if self.use_activation_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(blk, x, False, None, use_reentrant=False)
                else:
                    x = blk(x)
        q = self.query_tokens.repeat(len(x), 1, 1)
        q = self.cross_attention_block(q, x)
        return q


class AttentiveClassifier(nn.Module):
    """Attentive Classifier"""

    def __init__(
        self,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        num_classes=1000,
        complete_block=True,
        use_activation_checkpointing=False,
    ):
        super().__init__()
        self.pooler = AttentivePooler(
            num_queries=1,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            depth=depth,
            norm_layer=norm_layer,
            init_std=init_std,
            qkv_bias=qkv_bias,
            complete_block=complete_block,
            use_activation_checkpointing=use_activation_checkpointing,
        )
        self.linear = nn.Linear(embed_dim, num_classes, bias=True)

    def forward(self, x):
        x = self.pooler(x).squeeze(1)
        x = self.linear(x)
        return x


class AttentiveRegressor(nn.Module):
    """Attentive Regressor for continuous value prediction (e.g., skill scores)"""

    def __init__(
        self,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        num_outputs=1,
        complete_block=True,
        use_activation_checkpointing=False,
    ):
        super().__init__()
        self.pooler = AttentivePooler(
            num_queries=1,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            depth=depth,
            norm_layer=norm_layer,
            init_std=init_std,
            qkv_bias=qkv_bias,
            complete_block=complete_block,
            use_activation_checkpointing=use_activation_checkpointing,
        )
        self.linear = nn.Linear(embed_dim, num_outputs, bias=True)

    def forward(self, x):
        x = self.pooler(x).squeeze(1)
        x = self.linear(x)
        return x


class AttentiveHorizonRegressor(nn.Module):
    """Joint horizon-state classification and normalized regression head.

    Output layout follows the reference surgical anticipation implementation:
    [present logits for all targets,
     outside_horizon logits for all targets,
     inside_horizon logits for all targets,
     normalized regression for all targets]
    """

    def __init__(
        self,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        num_outputs=1,
        num_horizon_states=3,
        complete_block=True,
        use_activation_checkpointing=False,
    ):
        super().__init__()
        self.pooler = AttentivePooler(
            num_queries=1,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            depth=depth,
            norm_layer=norm_layer,
            init_std=init_std,
            qkv_bias=qkv_bias,
            complete_block=complete_block,
            use_activation_checkpointing=use_activation_checkpointing,
        )
        self.output_head = nn.Linear(
            embed_dim,
            num_outputs * (num_horizon_states + 1),
            bias=True,
        )
        self.num_outputs = num_outputs
        self.num_horizon_states = num_horizon_states

    def forward(self, x):
        x = self.pooler(x).squeeze(1)
        return self.output_head(x)

    def split_outputs(self, output):
        cls_dim = self.num_outputs * self.num_horizon_states
        state_logits = output[:, :cls_dim].reshape(-1, self.num_horizon_states, self.num_outputs)
        regression = output[:, cls_dim:]
        return regression, state_logits

    def load_state_dict(self, state_dict, strict=True):
        if (
            "regression_head.weight" in state_dict
            and "state_head.weight" in state_dict
            and "output_head.weight" not in state_dict
        ):
            state_weight = state_dict["state_head.weight"].view(
                self.num_outputs,
                self.num_horizon_states,
                -1,
            )
            state_bias = state_dict["state_head.bias"].view(
                self.num_outputs,
                self.num_horizon_states,
            )

            # Legacy checkpoints stored logits in target-major order:
            # [t0_present, t0_outside, t0_inside, t1_present, ...].
            # The new layout is class-major to match the reference code:
            # [present_all, outside_all, inside_all, reg_all].
            state_weight = state_weight.permute(1, 0, 2).reshape(
                self.num_outputs * self.num_horizon_states,
                -1,
            )
            state_bias = state_bias.permute(1, 0).reshape(
                self.num_outputs * self.num_horizon_states
            )

            converted_state_dict = {
                k: v
                for k, v in state_dict.items()
                if not k.startswith("regression_head.") and not k.startswith("state_head.")
            }
            converted_state_dict["output_head.weight"] = torch.cat(
                [state_weight, state_dict["regression_head.weight"]],
                dim=0,
            )
            converted_state_dict["output_head.bias"] = torch.cat(
                [state_bias, state_dict["regression_head.bias"]],
                dim=0,
            )
            state_dict = converted_state_dict

        return super().load_state_dict(state_dict, strict=strict)
