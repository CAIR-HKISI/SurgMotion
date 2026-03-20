import os
from sympy import N
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional, List
import torch.utils.checkpoint as checkpoint

from einops import rearrange
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights

import math

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from _mamba.mamba_ssm.modules.mamba_simple import Mamba
from _mamba.mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn

from video_sm.models.positional_encoding import PositionalEncoding

MODEL_PATH = '/data/tqy/endomamba_pretrain/'
_MODELS = {
    "videomamba_s16_in1k": os.path.join(MODEL_PATH, "endomamba_small_b48_seqlen16_withteacher_MIX12/checkpoint-499.pth"),
}

class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection.

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

        # Store bimamba flag for reshaping in forward pass
        self.bimamba = mixer_cls.keywords.get('bimamba', True)  # Default to True if not specified

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None,
        use_checkpoint=False
    ):
        """
        Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (residual + self.drop_path(hidden_states)) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states if residual is None else self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        if use_checkpoint:
            hidden_states = checkpoint.checkpoint(self.mixer, hidden_states, inference_params)
        else:
            hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    layer_idx=None,
    bimamba=True,
    device=None,
    dtype=None,
    return_last_state=False
):
    """
    Create a single Block with specified configuration.

    Args:
        d_model (int): Embedding dimension.
        ssm_cfg (dict, optional): Configuration for SSM.
        norm_epsilon (float, optional): Epsilon for normalization.
        drop_path (float, optional): Drop path rate.
        rms_norm (bool, optional): Use RMSNorm instead of LayerNorm.
        residual_in_fp32 (bool, optional): Use FP32 for residual connection.
        fused_add_norm (bool, optional): Fuse add and norm operations.
        bimamba (bool, optional): Use bimamba or mamba.
        device (torch.device, optional): Device for the mixer.
        dtype (torch.dtype, optional): Data type for the mixer.
        return_last_state (bool, optional): Whether to return the last state.

    Returns:
        Block: Configured block.
    """
    factory_kwargs = {"device": device, "dtype": dtype}
    if ssm_cfg is None:
        ssm_cfg = {}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba=bimamba, **ssm_cfg,
                        return_last_state=return_last_state, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# Weight Initialization Functions
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


# Patch Embedding Module
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, kernel_size=1, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.tubelet_size = kernel_size

        self.proj = nn.Conv3d(
            in_chans, embed_dim, 
            kernel_size=(kernel_size, patch_size[0], patch_size[1]),
            stride=(kernel_size, patch_size[0], patch_size[1])
        )

    def forward(self, x):
        x = self.proj(x)
        return x

# VisionMamba Model
class EndoMamba(nn.Module):
    def __init__(
            self, 
            img_size=224, 
            patch_size=16, 
            depth=24, 
            embed_dim=192, 
            channels=3, 
            num_classes=1000,
            drop_rate=0.,
            drop_path_rate=0.1,
            ssm_cfg=None, 
            norm_epsilon=1e-5, 
            initializer_cfg=None,
            fused_add_norm=True,
            rms_norm=True, 
            residual_in_fp32=True,
            # video
            kernel_size=1, 
            fc_drop_rate=0., 
            device=None,
            dtype=None,
            # checkpoint
            use_checkpoint=False,
            checkpoint_num=0,
            return_last_state=False,
            # new parameter to specify split between spatial and temporal layers
            num_spatial_layers=None,  # If None, default to depth // 2
            with_cls_token=True,
            only_cls_token=False,
            with_head=True,
            **kwargs
        ):
        """
        VisionMamba model with split spatial and temporal processing layers.

        Args:
            num_spatial_layers (int): Number of layers to process spatial information. Defaults to depth // 2.
            All other args are same as before.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
        print(f'Use checkpoint: {use_checkpoint}')
        print(f'Checkpoint number: {checkpoint_num}')
        
        self.intermediate_features = None
        
        self.img_size = img_size
        self.patch_size = patch_size

        # Pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, 
            kernel_size=kernel_size,
            in_chans=channels, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.with_cls_token = with_cls_token
        if with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.only_cls_token = only_cls_token and with_cls_token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches +1, self.embed_dim))
        self.temporal_pos_embedding = PositionalEncoding(embed_dim, 8192, device=device)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.with_head = with_head
        if self.with_head:
            self.head_drop = nn.Dropout(fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
            self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr

        # Determine number of spatial layers
        if num_spatial_layers is None:
            num_spatial_layers = depth // 2  # Default split

        self.layers = nn.ModuleList()
        self.temporal_layer_indices = []
        for i in range(depth):
            if i < num_spatial_layers:
                # Spatial layers with bimamba=True
                current_bimamba = True
            else:
                # Temporal layers with bimamba=False
                current_bimamba = False
                self.temporal_layer_indices.append(i)  # Keep track of temporal layers
            block = create_block(
                embed_dim,
                ssm_cfg=ssm_cfg,
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm,
                layer_idx=i,
                drop_path=inter_dpr[i],
                bimamba=current_bimamba,
                device=device,
                dtype=dtype,
                return_last_state=return_last_state
            )
            self.layers.append(block)

        # Output head normalization
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(embed_dim, eps=norm_epsilon, **factory_kwargs)

        # Initialize weights
        self.apply(segm_init_weights)
        if self.with_head:
            self.head.apply(segm_init_weights)
        trunc_normal_(self.pos_embed, std=.02)

        # Mamba-specific initialization
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

        self.return_last_state = return_last_state
        
    def get_lr_params(self, base_lr, factor=1, weight_decay=1e-5):
        params = [
            {"params": self.patch_embed.parameters(), "lr_scale": base_lr * factor, "lr_scale":factor, "weight_decay":weight_decay},
            # {"params": self.cls_token, "lr": base_lr * factor},
            {"params": self.pos_embed, "lr": base_lr * factor, "lr_scale":factor, "weight_decay":weight_decay},
            # {"params": self.temporal_pos_embedding, "lr": base_lr * factor},
            # {"params": self.head.parameters(), "lr": base_lr},
        ]
        for layer in self.layers:
            params.append({"params": layer.parameters(), "lr": base_lr * factor, "lr_scale":factor, "weight_decay":weight_decay})
        if self.with_cls_token:
            params.append({"params": self.cls_token, "lr": base_lr, "lr_scale":1, "weight_decay":0})
        if self.with_head:
            params.append({"params": self.head.parameters(), "lr": base_lr, "lr_scale":1, "weight_decay":weight_decay})
        return params
    
    def get_num_layers(self):
        return len(self.layers) 

    @torch.jit.ignore
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, x, inference_params: Optional[List[Optional[Tensor]]] = None):
        x = self.patch_embed(x)
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)  # (B*T, N, C)

        if self.with_cls_token:
            cls_token = self.cls_token.expand(B * T, -1, -1)  # (B*T, 1, C) stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_token, x), dim=1)  # (B*T, N+1, C)

        x = x + self.pos_embed

        if self.with_cls_token:
            cls_tokens = x[:B * T, :1, :]
            x = x[:, 1:]

        x = rearrange(x, '(b t) n c -> (b n) t c', b=B, t=T)  # (B*N, T, C)

        if inference_params is not None:
            temporal_pos = self.temporal_pos_embedding.encoding[inference_params.seqlen_offset: inference_params.seqlen_offset + T, :].unsqueeze(0).to(x.device)
            x = x + temporal_pos  # (B*N, T, C) + (1, T, C)
        else:
            x = x + self.temporal_pos_embedding(x).unsqueeze(0).to(x.device)  # (B*N, T, C)

        x = rearrange(x, '(b n) t c -> b t n c', b=B, t=T)  # (B, T, N, C)
        if self.with_cls_token:
            cls_tokens = cls_tokens.view(B, T, 1, C) # (B, T, 1, C)
            x_split = torch.cat((cls_tokens, x), dim=2) 
            x = rearrange(x_split, 'b t n c -> b t n c')  # (B, T, N+1, C)

        x = self.pos_drop(x)
        
        residual = None
        hidden_states = x  # (B, N+1, T, C)
        
        self.intermediate_features = []
        
        # Iterate over all layers
        for idx, layer in enumerate(self.layers):
            if layer.bimamba:
                # continue
                # Spatial processing: reshape to (B*T, N, C)
                hidden_states = rearrange(hidden_states, 'b t n m -> (b t) n m')  # (B*T, N, M)
                current_inference_param = None  # Spatial blocks do not use inference_params
                if residual is not None:
                    residual = rearrange(residual, 'b t n m -> (b t) n m')  # (B*T, N, m)
            else:
                hidden_states = rearrange(hidden_states, 'b t n m -> b (t n) m')  # (B*N, T, M)
                # Retrieve the corresponding inference_param for this temporal block
                # temporal_layer_idx = self.temporal_layer_indices.index(idx)
                current_inference_param = inference_params
                if residual is not None:
                    residual = rearrange(residual, 'b t n m -> b (t n) m')  # (B*N, T, m)
                    
            hidden_states, residual = layer(hidden_states, residual, inference_params=current_inference_param)

            if self.return_last_state and not layer.bimamba:
                hidden_states, inference_params = hidden_states

            if layer.bimamba:
                # Reshape back to (B, N, T, C)
                hidden_states = rearrange(hidden_states, '(b t) n m -> b t n m', b=B, t=T)
                if residual is not None:
                    residual = rearrange(residual, '(b t) n m -> b t n m', b=B, t=T) 
            else:
                # Reshape back to (B, N, T, C)
                hidden_states = rearrange(hidden_states, 'b (t n) m -> b t n m', n=x.shape[2], t=T)
                if residual is not None:
                    residual = rearrange(residual, 'b (t n) m -> b t n m', n=x.shape[2], t=T)
                    
            if idx == len(self.layers) // 3 or idx == 2 * len(self.layers) // 3 or idx == len(self.layers) - 1:
                self.intermediate_features.append(hidden_states)

        # # Final normalization
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        # Reshape back to (B, N, T, C) for final pooling
        # hidden_states = rearrange(hidden_states, 'b n t c -> b t n c')

        # Average over the spatial dimension (N+1*T)
        # frame_features = rearrange(hidden_states, 'b t n c -> b (t n) c', n=x.shape[2], t=T)  # Shape: (B, T*N, C)
        
        return hidden_states, inference_params

    def forward(self, x, inference_params: Optional[List[Optional[Tensor]]] = None):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor of shape (B, C, T, H, W).
            inference_params (List[Optional[Tensor]]): A list of inference parameters for temporal blocks.
                                                     The length should be equal to the number of temporal layers.
                                                     Each entry corresponds to a temporal block's cache.

        Returns:
            Tensor: Output logits of shape (B, T, num_classes).
            Optional inference_params if provided.
        """
        x, inference_params = self.forward_features(x, inference_params)
        if self.with_head:
            # x = x.mean(dim=2)
            if self.only_cls_token:
                x = x[:, :, 0, :]  # take only cls token
            else:
                x = x.mean(dim=2)  #  (B, T, N, C) --> (B, T, C)
            # x = rearrange(x, 'b t n c -> b (t n) c', n=x.shape[2], t=T)  # Shape: (B, T*N, C)
            x = self.head(self.head_drop(x))
        if inference_params is not None:
            return x, inference_params
        else:
            return x
        
    def get_features(self):
        return self.intermediate_features
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}


def inflate_weight(weight_2d, time_dim, center=True):
    """
    Inflate 2D weights to 3D by adding a temporal dimension.

    Args:
        weight_2d (Tensor): 2D weights of shape (out_channels, in_channels, H, W).
        time_dim (int): Size of the temporal dimension.
        center (bool): Whether to place the 2D weights at the center of the temporal dimension.

    Returns:
        Tensor: 3D weights of shape (out_channels, in_channels, T, H, W).
    """
    print(f'Init center: {center}')
    if center:
        weight_3d = torch.zeros(*weight_2d.shape, time_dim)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    else:
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim
    return weight_3d


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
    state_dict = {k[len('encoder.'):]: v for k, v in state_dict.items() if k.startswith('encoder.')}

    for k in list(state_dict.keys()):
        if k in state_dict_3d.keys() and state_dict[k].shape != state_dict_3d[k].shape:
            if len(state_dict_3d[k].shape) <= 3:
                print(f'Ignore: {k}')
                del state_dict[k]
                continue
            print(f'Inflate: {k}, {state_dict[k].shape} => {state_dict_3d[k].shape}')
            time_dim = state_dict_3d[k].shape[2]
            try:
                state_dict[k] = inflate_weight(state_dict[k], time_dim, center=center)
            except:
                print(f"Failed to inflate {k} from {state_dict[k].shape} to {state_dict_3d[k].shape}")
                del state_dict[k]

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
def endomamba_tiny(pretrained=False, **kwargs):
    """
    Create a tiny VisionMamba model.

    Args:
        pretrained (bool): Whether to load pretrained weights.
        **kwargs: Additional arguments.

    Returns:
        VisionMamba: The VisionMamba tiny model.
    """
    model = EndoMamba(
        patch_size=16, 
        embed_dim=192, 
        depth=24, 
        rms_norm=True, 
        residual_in_fp32=True, 
        fused_add_norm=True, 
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
def endomamba_small(pretrained=False, **kwargs):
    """
    Create a small VisionMamba model.

    Args:
        pretrained (bool): Whether to load pretrained weights.
        **kwargs: Additional arguments.

    Returns:
        VisionMamba: The VisionMamba small model.
    """
    model = EndoMamba(
        embed_dim=384, 
        depth=24, 
        rms_norm=True, 
        residual_in_fp32=True, 
        fused_add_norm=True, 
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
def endomamba_middle(pretrained=False, **kwargs):
    """
    Create a middle-sized VisionMamba model with support for recurrent inference.

    Args:
        pretrained (bool): Whether to load pretrained weights.
        **kwargs: Additional arguments.

    Returns:
        VisionMamba: The VisionMamba middle model.
    """
    model = EndoMamba(
        patch_size=16, 
        embed_dim=576, 
        depth=32, 
        rms_norm=True, 
        residual_in_fp32=True, 
        fused_add_norm=True, 
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        print('Loading pretrained weights...')
        state_dict = torch.load(_MODELS["videomamba_m16_in1k"], map_location='cpu')
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
        load_state_dict(model, state_dict, center=True)
    return model


# Example usage and FLOPs calculation
if __name__ == '__main__':
    import time
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table
    import numpy as np

    seed = 4217
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    num_frames = 8
    img_size = 224

    # To evaluate GFLOPs, please set `rms_norm=False` and `fused_add_norm=False`
    model = endomamba_middle().cuda()
    flops = FlopCountAnalysis(model, torch.rand(1, 3, num_frames, img_size, img_size).cuda())
    s = time.time()
    print(flop_count_table(flops, max_depth=1))
    print(time.time()-s)
