from pathlib import Path

from setuptools import find_packages, setup

NAME = "surgmotion"
VERSION = "0.1.0"
DESCRIPTION = (
    "SurgMotion: A Video-Native Foundation Model for "
    "Universal Understanding of Surgical Videos."
)
URL = "https://github.com/CAIR-HKISI/SurgMotion"

_ROOT = Path(__file__).resolve().parent


def get_requirements():
    lines: list[str] = []
    for raw in (_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


if __name__ == "__main__":
    setup(
        name=NAME,
        version=VERSION,
        description=DESCRIPTION,
        url=URL,
        python_requires=">=3.10",
        packages=find_packages(exclude=["tests", "tests.*"]),
        install_requires=get_requirements(),
    )
