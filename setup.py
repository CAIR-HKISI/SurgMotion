# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from setuptools import setup

NAME = "vjepa2"
VERSION = "0.0.1"
DESCRIPTION = "PyTorch code and models for V-JEPA 2."
URL = "https://github.com/facebookresearch/vjepa2"

_ROOT = Path(__file__).resolve().parent


def _expand_requirements(req_file: Path) -> list[str]:
    """Resolve `-r other.txt` includes for pip-style requirement files."""
    lines: list[str] = []
    for raw in req_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r "):
            sub = (req_file.parent / line[3:].strip()).resolve()
            lines.extend(_expand_requirements(sub))
        else:
            lines.append(line)
    return lines


def get_requirements():
    return _expand_requirements(_ROOT / "requirements.txt")


if __name__ == "__main__":
    setup(
        name=NAME,
        version=VERSION,
        description=DESCRIPTION,
        url=URL,
        python_requires=">=3.10",
        install_requires=get_requirements(),
    )
