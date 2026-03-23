import json
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.depth import SurgicalDepthHead


def main():
    torch.manual_seed(0)

    head = SurgicalDepthHead(
        embed_dim=192,
        decoder_channels=(64, 32),
        patch_size=16,
        target_size=(64, 64),
        num_feature_levels=1,
        activation="softplus",
    ).cpu()
    head.eval()

    with torch.no_grad():
        features = torch.randn(2, 16, 192, dtype=torch.float32)
        outputs = head(features)

    expected_shape = (2, 1, 64, 64)
    if tuple(outputs.shape) != expected_shape:
        raise ValueError(f"Expected output shape={expected_shape}, got {tuple(outputs.shape)}")

    summary = {
        "import_ok": True,
        "head_class": head.__class__.__name__,
        "output_shape": tuple(outputs.shape),
        "output_dtype": str(outputs.dtype),
        "output_device": str(outputs.device),
    }
    print("DEPTH_HEAD_FORWARD_SMOKE_PASS", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
