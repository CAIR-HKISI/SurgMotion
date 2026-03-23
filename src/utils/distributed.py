# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

import torch
import torch.distributed as dist

from src.utils.logging import get_logger

logger = get_logger()


# def init_distributed(port=37129, rank_and_world_size=(None, None)):
#     # try to set all environment variables to avoid triggering a segfault
#     # environment variables can be reallocated during the execution of torch.distributed.init_process_group
#     # the idea is a race condition may trigger if init_progress_group is modifying an environment variable at
#     # the same time as Python, so we try to set all environs before initializing distributed
#     if "SLURM_JOB_ID" in os.environ:
#         # Use the slurm_tmpdir (if it exists) instead of /tmp
#         tmpdir = Path(f"/scratch/slurm_tmpdir/{os.environ['SLURM_JOB_ID']}")
#         if tmpdir.exists():
#             os.environ["TMPDIR"] = str(tmpdir)

#     if dist.is_available() and dist.is_initialized():
#         return dist.get_world_size(), dist.get_rank()

#     rank, world_size = rank_and_world_size
#     os.environ["MASTER_ADDR"] = "localhost"

#     if (rank is None) or (world_size is None):
#         try:
#             world_size = int(os.environ["SLURM_NTASKS"])
#             rank = int(os.environ["SLURM_PROCID"])
#             os.environ["MASTER_ADDR"] = os.environ["HOSTNAME"]
#         except Exception:
#             logger.info("SLURM vars not set (distributed training not available)")
#             world_size, rank = 1, 0
#             return world_size, rank

#     try:
#         os.environ["MASTER_PORT"] = str(port)
#         torch.distributed.init_process_group(backend="nccl", world_size=world_size, rank=rank)
#     except Exception as e:
#         world_size, rank = 1, 0
#         logger.info(f"Rank: {rank}. Distributed training not available {e}")

#     return world_size, rank


def init_distributed(port=37129, rank_and_world_size=(None, None)):
    if "SLURM_JOB_ID" in os.environ:
        tmpdir = Path(f"/scratch/slurm_tmpdir/{os.environ['SLURM_JOB_ID']}")
        if tmpdir.exists():
            os.environ["TMPDIR"] = str(tmpdir)

        if port == 37129:
            port = 20000 + (int(os.environ["SLURM_JOB_ID"]) % 40000)

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()

    rank, world_size = rank_and_world_size
    os.environ.setdefault("MASTER_ADDR", "localhost")

    if (rank is None) or (world_size is None):
        try:
            world_size = int(os.environ["SLURM_NTASKS"])
            rank = int(os.environ["SLURM_PROCID"])

            if os.environ.get("MASTER_ADDR") in {"localhost", "127.0.0.1"}:
                if "SLURM_LAUNCH_NODE_IPADDR" in os.environ:
                    os.environ["MASTER_ADDR"] = os.environ["SLURM_LAUNCH_NODE_IPADDR"]
                elif "SLURM_JOB_NODELIST" in os.environ:
                    import subprocess

                    nodes = (
                        subprocess.check_output(["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]])
                        .decode()
                        .split()
                    )
                    if len(nodes) > 0:
                        os.environ["MASTER_ADDR"] = nodes[0]
                    else:
                        os.environ["MASTER_ADDR"] = os.environ.get("HOSTNAME", "localhost")
                else:
                    os.environ["MASTER_ADDR"] = os.environ.get("HOSTNAME", "localhost")
        except Exception as e:
            logger.info(f"SLURM vars not set or error: {e}. Falling back to non-distributed.")
            world_size, rank = 1, 0
            return world_size, rank

    if world_size == 1:
        logger.info("Single process mode (world_size=1), skipping distributed initialization")
        return world_size, rank

    try:
        if "MASTER_PORT" in os.environ:
            try:
                port = int(os.environ["MASTER_PORT"])
            except Exception:
                pass
        os.environ["MASTER_PORT"] = str(port)
        logger.info(
            "Initializing distributed: "
            f"MASTER_ADDR={os.environ['MASTER_ADDR']}, MASTER_PORT={port}, rank={rank}, world_size={world_size}"
        )
        dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
    except Exception as e:
        logger.error(f"Failed to initialize process group: {e}")
        world_size, rank = 1, 0

    return world_size, rank

class AllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            x = x.contiguous()
            outputs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(outputs, x)
            return torch.cat(outputs, 0)
        return x

    @staticmethod
    def backward(ctx, grads):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            s = (grads.shape[0] // dist.get_world_size()) * dist.get_rank()
            e = (grads.shape[0] // dist.get_world_size()) * (dist.get_rank() + 1)
            grads = grads.contiguous()
            dist.all_reduce(grads)
            return grads[s:e]
        return grads


class AllReduceSum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            x = x.contiguous()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads


class AllReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads
