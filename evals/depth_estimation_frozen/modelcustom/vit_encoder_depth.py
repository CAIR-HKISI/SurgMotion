import logging

import torch

import src.models.vision_transformer as vit

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def init_module(
    resolution: int,
    checkpoint: str,
    model_kwargs: dict,
    wrapper_kwargs: dict,
    **kwargs,
):
    if checkpoint:
        logger.info(f"Loading pretrained model from {checkpoint}")
        checkpoint = torch.load(checkpoint, map_location="cpu")
    else:
        logger.info("No checkpoint provided; using randomly initialized encoder for smoke execution")
        checkpoint = None

    enc_kwargs = dict(model_kwargs["encoder"])
    enc_ckp_key = enc_kwargs.pop("checkpoint_key", "target_encoder")
    enc_model_name = enc_kwargs.pop("model_name")

    img_as_video_nframes = wrapper_kwargs.get("img_as_video_nframes", enc_kwargs.get("num_frames", 2))
    out_layers = wrapper_kwargs.get("out_layers")
    if out_layers is not None:
        enc_kwargs["out_layers"] = out_layers
    enc_kwargs.pop("num_frames", None)

    model = vit.__dict__[enc_model_name](
        input_size=resolution,
        num_frames=img_as_video_nframes,
        **enc_kwargs,
    )

    def forward_prehook(module, inputs):
        x = inputs[0]
        if x.ndim != 4:
            return inputs
        x = x.unsqueeze(2).repeat(1, 1, img_as_video_nframes, 1, 1)
        return x

    model.register_forward_pre_hook(forward_prehook)

    if checkpoint is not None:
        if enc_ckp_key in checkpoint:
            pretrained_dict = checkpoint[enc_ckp_key]
        else:
            pretrained_dict = checkpoint
        pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
        pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
        model_state = model.state_dict()
        for k, v in model_state.items():
            if k not in pretrained_dict:
                logger.info(f'Key "{k}" could not be found in loaded state dict')
            elif pretrained_dict[k].shape != v.shape:
                logger.info(f'Key "{k}" has mismatched shape ({pretrained_dict[k].shape} vs {v.shape})')
                pretrained_dict[k] = v
        msg = model.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"Loaded pretrained encoder with msg: {msg}")
        del checkpoint
    return model
