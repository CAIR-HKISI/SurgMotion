import importlib
import logging

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def init_module(
    module_name,
    device,
    resolution,
    checkpoint,
    model_kwargs,
    wrapper_kwargs,
):
    model = (
        importlib.import_module(f"{module_name}")
        .init_module(
            resolution=resolution,
            checkpoint=checkpoint,
            model_kwargs=model_kwargs,
            wrapper_kwargs=wrapper_kwargs,
        )
        .to(device)
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(model)
    return model
