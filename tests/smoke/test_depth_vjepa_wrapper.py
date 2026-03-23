import json
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.depth_estimation_frozen.modelcustom import init_module


def main():
    torch.manual_seed(0)

    model = init_module(
        resolution=224,
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
            "out_layers": [0, 1],
        },
    )
    model.eval()

    with torch.no_grad():
        images = torch.randn(2, 3, 224, 224, dtype=torch.float32)
        outputs = model(images)

    if not isinstance(outputs, list):
        raise TypeError(f"Expected list output when out_layers is set, got {type(outputs).__name__}")
    if len(outputs) != 2:
        raise ValueError(f"Expected 2 layer outputs, got {len(outputs)}")

    expected_shape = (2, 196, 192)
    output_shapes = [tuple(out.shape) for out in outputs]
    if any(shape != expected_shape for shape in output_shapes):
        raise ValueError(f"Expected all output shapes={expected_shape}, got {output_shapes}")

    summary = {
        "import_ok": True,
        "duplicate_num_frames_ok": True,
        "model_class": model.__class__.__name__,
        "output_kind": type(outputs).__name__,
        "output_shapes": output_shapes,
    }
    print("DEPTH_VJEPA_WRAPPER_SMOKE_PASS", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
