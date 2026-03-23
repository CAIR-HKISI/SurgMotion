# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import subprocess
import sys

import torch


def gpu_timer(closure, log_timings=True):
    """Helper to time gpu-time to execute closure()"""
    log_timings = log_timings and torch.cuda.is_available()

    elapsed_time = -1.0
    if log_timings:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    result = closure()

    if log_timings:
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)

    return result, elapsed_time


LOG_FORMAT = "[%(levelname)-8s][%(asctime)s][%(name)-20s][%(funcName)-25s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name=None, force=False):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT, force=force)
    return logging.getLogger(name=name)


class CSVLogger(object):

    def __init__(self, fname, *argv, **kwargs):
        self.fname = fname
        self.types = []
        self.delim = kwargs.get("delim", ",")
        mode = kwargs.get("mode", "a")

        expected_cols = [v[1] for v in argv]
        for v in argv:
            self.types.append(v[0])

        needs_header = True
        if "a" in mode and os.path.exists(self.fname) and os.path.getsize(self.fname) > 0:
            with open(self.fname, "r") as f:
                first_line = f.readline().strip()
            existing_cols = first_line.split(self.delim) if first_line else []
            if existing_cols == expected_cols:
                needs_header = False
            else:
                base, ext = os.path.splitext(self.fname)
                ext = ext if ext else ".csv"
                i = 1
                while True:
                    candidate = f"{base}.v{i}{ext}"
                    if not os.path.exists(candidate):
                        self.fname = candidate
                        break
                    i += 1
                needs_header = True

        if needs_header:
            os.makedirs(os.path.dirname(self.fname) or ".", exist_ok=True)
            with open(self.fname, "w") as f:
                for i, col in enumerate(expected_cols, 1):
                    end = self.delim if i < len(expected_cols) else "\n"
                    print(col, end=end, file=f)

    def log(self, *argv):
        if len(argv) != len(self.types):
            raise ValueError(
                f"CSVLogger.log argument count mismatch: got {len(argv)} values but logger expects {len(self.types)}. "
                f"file='{self.fname}'"
            )
        with open(self.fname, "a") as f:
            for i, tv in enumerate(zip(self.types, argv), 1):
                end = self.delim if i < len(self.types) else "\n"
                print(tv[0] % tv[1], end=end, file=f)


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.max = float("-inf")
        self.min = float("inf")
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        try:
            self.max = max(val, self.max)
            self.min = min(val, self.min)
        except Exception:
            pass
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def jepa_rootpath():
    this_file = os.path.abspath(__file__)
    return "/".join(this_file.split("/")[:-3])


def git_information():
    jepa_root = jepa_rootpath()
    try:
        resp = (
            subprocess.check_output(["git", "-C", jepa_root, "rev-parse", "HEAD", "--abbrev-ref", "HEAD"])
            .decode("ascii")
            .strip()
        )
        commit, branch = resp.split("\n")
        return f"branch: {branch}\ncommit: {commit}\n"
    except Exception:
        return "unknown"
