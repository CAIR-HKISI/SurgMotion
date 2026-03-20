from regex import F
import torch
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from video_sm.models.endomamba import *
from _mamba.mamba_ssm.utils.generation import InferenceParams

# Define batch, sequence length, and feature dimension
batch, total_length, dim = 2, 8, 224 
split_num = 1
device = "cuda:0"

with torch.cuda.device(device):
    x = torch.randn(batch, 3, total_length, dim, dim).to(device) # Define input tensor x

    segments = torch.split(x, split_num, dim=2)  # Split x into segments along the length

    model = endomamba_small(
        num_classes=7, 
        pretrained=True, 
        return_last_state=True, 
        with_head=True,
    ).to(device)
    model.eval()
    
    infer_params = InferenceParams(max_seqlen=16, max_batch_size=1)
    
    # Parallel Computation
    y, _ = model(x, infer_params)
    
    infer_params = InferenceParams(max_seqlen=16, max_batch_size=1)
    
    # Recurrent Computation: Process each segment separately and collect results
    y_segments = []
    for segment in segments:
        y_segment, infer_params = model(segment.to(device), inference_params=infer_params)
        y_segments.append(y_segment)
        infer_params.seqlen_offset += 1

    # Concatenate the results along the length dimension
    y_ = torch.cat(y_segments, dim=1)

    # Check the error between the two outputs
    print("Error_max = ", (y - y_).abs().max())
    print("Error_mean = ", (y - y_).abs().mean())
    print("Error_std = ", (y - y_).abs().std())

    # Assert that the outputs are similar within a tolerance
    assert torch.allclose(y, y_, atol=1e-5), "Outputs are not equal!"
    print("\nThe outputs of recurrent and parallel modes are equivalent.")
    
# ─────────────────────────────────────────────────────────────
# 🔄 Optional: for training usage, disable return_last_state
# model = endomamba_small(
#     num_classes=7,
#     pretrained=True,
#     return_last_state=False,
#     with_head=True,
# ).to(device)
# y_train = model(x)  # Parallel mode only