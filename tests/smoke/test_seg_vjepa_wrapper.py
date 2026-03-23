import json
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.segmentation_frozen.modelcustom import init_module


def main():
    torch.manual_seed(0)

    model = init_module(
        resolution=64,
        checkpoint=None,
        model_kwargs={
            "encoder": {
                "model_name": "vit_tiny",
                "checkpoint_key": "target_encoder",
                "patch_size": 16,
                "tubelet_size": 2,
                "num_frames": 2,
                "uniform_power": False,
                "use_silu": False,
                "wide_silu": True,
                "use_sdpa": False,
                "use_rope": False,
            }
        },
        wrapper_kwargs={
            "img_as_video_nframes": 2,
        },
    )
    model.eval()

    with torch.no_grad():
        images = torch.randn(2, 3, 64, 64, dtype=torch.float32)
        outputs = model(images)

    expected_shape = (2, 16, 192)
    if not isinstance(outputs, torch.Tensor):
        raise TypeError(f"Expected Tensor output, got {type(outputs).__name__}")
    if tuple(outputs.shape) != expected_shape:
        raise ValueError(f"Expected output shape={expected_shape}, got {tuple(outputs.shape)}")

    summary = {
        "import_ok": True,
        "wrapper_class": model.__class__.__name__,
        "input_shape": (2, 3, 64, 64),
        "output_type": type(outputs).__name__,
        "output_shape": tuple(outputs.shape),
    }
    print("SEG_VJEPA_WRAPPER_SMOKE_PASS", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
