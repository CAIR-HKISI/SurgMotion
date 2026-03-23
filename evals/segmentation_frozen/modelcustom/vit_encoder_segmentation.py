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
    logger.info(f"Loading pretrained model from {checkpoint}")
    checkpoint_data = None
    if checkpoint is not None:
        checkpoint_data = torch.load(checkpoint, map_location="cpu")
    else:
        logger.info("No checkpoint provided; using randomly initialized segmentation encoder for smoke run.")

    enc_kwargs = dict(model_kwargs["encoder"])
    enc_ckp_key = enc_kwargs.pop("checkpoint_key")
    enc_model_name = enc_kwargs.pop("model_name")

    enc_num_frames = enc_kwargs.pop("num_frames", None)
    enc_kwargs.pop("img_size", None)
    enc_kwargs.pop("input_size", None)

    img_as_video_nframes = wrapper_kwargs.get("img_as_video_nframes", enc_num_frames or 2)
    out_layers = wrapper_kwargs.get("out_layers")
    if out_layers is not None:
        enc_kwargs["out_layers"] = out_layers

    model = vit.__dict__[enc_model_name](
        img_size=resolution,
        num_frames=img_as_video_nframes,
        **enc_kwargs,
    )

    def forward_prehook(module, inputs):
        x = inputs[0]
        if x.ndim != 4:
            return inputs
        x = x.unsqueeze(2).repeat(1, 1, img_as_video_nframes, 1, 1)
        return (x,)

    model.register_forward_pre_hook(forward_prehook)

    if checkpoint_data is None:
        pretrained_dict = {}
    elif enc_ckp_key in checkpoint_data:
        pretrained_dict = checkpoint_data[enc_ckp_key]
    else:
        pretrained_dict = checkpoint_data

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
    del checkpoint_data
    return model
