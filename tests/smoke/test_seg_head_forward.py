import json
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.segmentation import MaskFormerSegmentationHead


def main():
    torch.manual_seed(0)

    head = MaskFormerSegmentationHead(
        embed_dim=192,
        patch_size=16,
        target_size=64,
        num_feature_levels=1,
        hidden_dim=64,
        mask_dim=64,
        num_queries=1,
        num_decoder_layers=2,
        nheads=4,
        dim_feedforward=128,
        decoder_channels=(64, 32),
        return_aux=True,
    ).cpu()
    head.eval()

    with torch.no_grad():
        features = torch.randn(2, 16, 192, dtype=torch.float32)
        outputs = head(features)

    logits = outputs["logits"]
    aux_logits = outputs["aux_logits"]
    expected_shape = (2, 1, 64, 64)
    if tuple(logits.shape) != expected_shape:
        raise ValueError(f"Expected logits shape={expected_shape}, got {tuple(logits.shape)}")

    summary = {
        "import_ok": True,
        "head_class": head.__class__.__name__,
        "logits_shape": tuple(logits.shape),
        "aux_count": len(aux_logits),
        "logits_mean": round(float(logits.mean()), 6),
    }
    print("SEG_HEAD_FORWARD_SMOKE_PASS", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
