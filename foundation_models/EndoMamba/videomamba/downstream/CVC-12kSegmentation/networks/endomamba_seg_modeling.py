# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
import os
from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from einops import rearrange
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from video_sm.models.endomamba import EndoMamba, inflate_weight


logger = logging.getLogger(__name__)


MODEL_PATH = '/data/tqy/endomamba_pretrain/'
_MODELS = {
    "videomamba_s16_in1k": os.path.join(MODEL_PATH, "endomamba_small_b48_seqlen16_withteacher_MIX12/checkpoint-499.pth"),
}


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            # skip [8, 384, 14, 14]
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, hidden_size=768, n_skip=3, 
                 skip_channels=[512, 256, 64, 16], decoder_channels=(256, 128, 64, 16)):
        super().__init__()
        head_channels = 512
        self.conv_more = Conv2dReLU(
            hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        self.n_skip = n_skip
        self.skip_channels = skip_channels
        if self.n_skip != 0:
            skip_channels = self.skip_channels
            for i in range(4-self.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        if self.n_skip != 0:
            self.up = [nn.UpsamplingBilinear2d(scale_factor=2),
                    nn.UpsamplingBilinear2d(scale_factor=4),
                    nn.UpsamplingBilinear2d(scale_factor=8),]

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.n_skip) else None
            else:
                skip = None
            if skip is not None:
                skip = skip[:, :, 1:, :].contiguous().view(B, hidden, h, w)
                skip = self.up[i](skip)
            x = decoder_block(x, skip=skip)
        return x


class EndoMambaSeg(nn.Module):
    def __init__(self,
                 img_size=224, 
                 patch_size=16, 
                 embed_dim=192, 
                 depth=24, 
                 rms_norm=True, 
                 residual_in_fp32=True, 
                 fused_add_norm=True,  
                 num_classes=2, 
                 zero_head=False, vis=False,
                 # decoder configs
                 n_skip=3, 
                 skip_channels=[512, 256, 64, 16], 
                 decoder_channels=(256, 128, 64, 16),
                 **kwargs, ):
        super(EndoMambaSeg, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.encoder = EndoMamba(
                        patch_size=patch_size, 
                        embed_dim=embed_dim, 
                        depth=depth, 
                        # num_spatial_layers=depth,
                        rms_norm=rms_norm, 
                        residual_in_fp32=residual_in_fp32, 
                        fused_add_norm=fused_add_norm, 
                        with_head=False,
                    )
        total_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        print(f'Backbone trainable parameters: {total_params}')
        self.decoder = DecoderCup(
            hidden_size=embed_dim, 
            n_skip=n_skip, 
            skip_channels=skip_channels, 
            decoder_channels=decoder_channels)
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=num_classes,
            kernel_size=3,
        )

    def forward(self, x):
        # Only work with batchsize = 1 as the sequence length T is not fixed
        # if x.size()[1] == 1:
        #     x = x.repeat(1,3,1,1)
        x = self.encoder(x)  # (B, T, N, C)
        x = rearrange(x, 'b t n c -> (b t) n c')  # Shape: (B*T, N+1, C)
        x = x[:, 1:, :]  # Remove the cls token
        x = self.decoder(x, self.encoder.get_features())
        # print(x.shape)
        logits = self.segmentation_head(x)
        # print('logits', logits.shape); exit(0)
        return logits


def load_state_dict(model, state_dict, center=True):
    """
    Load a state dict into the model, handling 2D to 3D weight inflation if necessary.

    Args:
        model (nn.Module): The model to load the state dict into.
        state_dict (dict): The state dictionary.
        center (bool): Whether to center the inflated weights.
    """
    state_dict_3d = model.state_dict()
    
    # Remove the 'encoder.' prefix if present
    # state_dict = {k[len('encoder.'):]: v for k, v in state_dict.items() if k.startswith('encoder.')}

    for k in list(state_dict.keys()):
        if k in state_dict_3d.keys() and state_dict[k].shape != state_dict_3d[k].shape:
            if len(state_dict_3d[k].shape) <= 3:
                print(f'Ignore: {k}')
                del state_dict[k]
                continue
            print(f'Inflate: {k}, {state_dict[k].shape} => {state_dict_3d[k].shape}')
            time_dim = state_dict_3d[k].shape[2]
            state_dict[k] = inflate_weight(state_dict[k], time_dim, center=center)

    # Handle positional embeddings
    if "temporal_pos_embedding" in state_dict.keys() and "temporal_pos_embedding" in state_dict_3d.keys():
        k = "temporal_pos_embedding"
        if state_dict[k].shape != state_dict_3d[k].shape:
            print(f"Size mismatch for {k}: {state_dict[k].shape} != {state_dict_3d[k].shape}, deleting it.")
            del state_dict[k] 
    if "pos_embed" in state_dict.keys() and "pos_embed" in state_dict_3d.keys():
        k = "pos_embed"
        if state_dict[k].shape != state_dict_3d[k].shape:
                state_dict[k] = state_dict[k][:, :state_dict_3d[k].shape[1], :]
            
    # Remove head weights
    if 'head.weight' in state_dict:
        del state_dict['head.weight']
    if 'head.bias' in state_dict:
        del state_dict['head.bias']
 
    # Load state dict
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    

@register_model
def endomambaseg_tiny(pretrained=False, **kwargs):
    """
    Create a tiny VisionMamba model.

    Args:
        pretrained (bool): Whether to load pretrained weights.
        **kwargs: Additional arguments.

    Returns:
        VisionMamba: The VisionMamba tiny model.
    """
    model = EndoMambaSeg(
        patch_size=16, 
        embed_dim=192, 
        depth=24, 
        rms_norm=True, 
        residual_in_fp32=True, 
        fused_add_norm=True, 
        n_skip=0, 
        skip_channels=[512, 256, 64, 16], 
        decoder_channels=(256, 128, 64, 16),
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        print('Loading pretrained weights...')
        state_dict = torch.load(_MODELS["videomamba_t16_in1k"], map_location='cpu')
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
        load_state_dict(model, state_dict, center=True)
    return model


@register_model
def endomambaseg_small(pretrained=False, **kwargs):
    """
    Create a small VisionMamba model.

    Args:
        pretrained (bool): Whether to load pretrained weights.
        **kwargs: Additional arguments.

    Returns:
        VisionMamba: The VisionMamba small model.
    """
    model = EndoMambaSeg(
        patch_size=16, 
        embed_dim=384, 
        depth=24, 
        rms_norm=True, 
        residual_in_fp32=True, 
        fused_add_norm=True, 
        skip_channels=[384, 384, 384, 384], 
        decoder_channels=(256, 128, 64, 16),
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        print('Loading pretrained weights...')
        state_dict = torch.load(_MODELS["videomamba_s16_in1k"], map_location='cpu')
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
        load_state_dict(model, state_dict, center=True)
    return model


@register_model
def endomambaseg_middle(pretrained=False, **kwargs):
    """
    Create a middle-sized VisionMamba model with support for recurrent inference.

    Args:
        pretrained (bool): Whether to load pretrained weights.
        **kwargs: Additional arguments.

    Returns:
        VisionMamba: The VisionMamba middle model.
    """
    model = EndoMambaSeg(
        patch_size=16, 
        embed_dim=576, 
        depth=32, 
        rms_norm=True, 
        residual_in_fp32=True, 
        fused_add_norm=True, 
        n_skip=0, 
        skip_channels=[512, 256, 64, 16], 
        decoder_channels=(256, 128, 64, 16),
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        print('Loading pretrained weights...')
        state_dict = torch.load(_MODELS["videomamba_m16_in1k"], map_location='cpu')
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
        load_state_dict(model, state_dict, center=True)
    return model


if __name__ == '__main__':
    import time
    import numpy as np
    batch, total_length, dim = 1, 8, 224  # Set length as 64, matching the requirement
    split_num = 1
    device = "cuda:0"
    with torch.cuda.device(device):
        x = torch.randn(total_length, 3, dim, dim).to(device)
        x = x.unsqueeze(0).permute(0, 2, 1, 3, 4)
        # x = torch.randn(batch, 3, total_length, dim, dim).to(device)

        model = endomambaseg_small(pretrained=True, 
                                   num_classes=1,
                                   
                                   ).to(device)
        
        y = model(x)
        
        print(y.size())
        
        pass
        
