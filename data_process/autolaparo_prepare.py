#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoLaparo preprocessing script
--------------------------------
Data layout (expected):
  - Frames (already extracted, e.g. 1 fps): ``<frames_root>/XX/XX_XXXXXXXX.jpg``
    where ``XX`` is zero-padded case id (01–21).
  - Per-frame phase labels: ``<labels_root>/label_XX.txt`` (whitespace-separated columns:
    ``Frame``, ``Phase`` with ``Phase`` in 1–7).

This script:
1. Optionally extracts frames from videos via ``videos_to_frames()`` (commented out by default;
   frames are usually pre-extracted for this dataset).
2. For each case, writes ``<output_dir>/clip_infos/XX.txt`` listing every frame path (one per line),
   paths use forward slashes (relative / as stored under project root, same style as PolypDiag).
3. Builds metadata rows (one row per case / clip): reads labels, maps phase ``1–7`` → ``0–6``,
   sets ``label`` / ``label_name`` to the **majority phase** among label rows whose frame file
   exists under ``frames_root`` (ties broken by smallest phase id). If no valid labeled frame exists,
   the case is skipped with a warning.
4. Splits by case id: train 1–10, val 11–14, test 15–21.
5. Writes ``train_metadata.csv``, ``val_metadata.csv``, ``test_metadata.csv`` under ``output_dir``.

Output CSV columns:
  ``Index``, ``clip_path``, ``label``, ``label_name``, ``case_id``, ``clip_idx``

  - ``clip_path``: path to the case’s ``clip_infos/XX.txt``.
  - ``case_id``: integer video / case number (1–21).
  - ``clip_idx``: always ``0`` (one clip per case).
"""

from __future__ import annotations

import argparse
import subprocess
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

# Train / val / test by case id (video number)
TRAIN_CASE_IDS = list(range(1, 11))  # 1–10
VAL_CASE_IDS = list(range(11, 15))  # 11–14
TEST_CASE_IDS = list(range(15, 22))  # 15–21

PHASE_ID_TO_NAME: Dict[int, str] = {
    0: "Preparation",
    1: "Dividing Ligament and Peritoneum",
    2: "Dividing Uterine Vessels and Ligament",
    3: "Transecting the Vagina",
    4: "Specimen Removal",
    5: "Suturing",
    6: "Washing",
}


def split_for_case(case_id: int) -> Optional[str]:
    """Return ``train``, ``val``, ``test``, or ``None`` if case id is outside defined splits."""
    if case_id in TRAIN_CASE_IDS:
        return "train"
    if case_id in VAL_CASE_IDS:
        return "val"
    if case_id in TEST_CASE_IDS:
        return "test"
    return None


def videos_to_frames(
    input_path: Path,
    output_path: Path,
    fps: int = 1,
    debug: bool = False,
) -> None:
    """
    Extract frames from all ``*.mp4`` under ``input_path`` into ``output_path``:
    ``<video_stem>.mp4`` → ``output_path/<video_stem>/<video_stem>_%05d.jpg``.

    AutoLaparo releases often ship with frames already at 1 fps; keep this for pipeline parity
    and run only if you have raw videos and need extraction.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    video_files = sorted(input_path.glob("*.mp4"))

    if not video_files:
        print(f"⚠️ No mp4 videos found in {input_path}.")
        return

    print(f"\n🎞️ Found {len(video_files)} videos, extracting frames...\n")
    failed: List[str] = []

    for vid_path in tqdm(video_files, desc="Extracting frames"):
        vid_id = vid_path.stem
        out_folder = output_path / vid_id
        out_folder.mkdir(parents=True, exist_ok=True)
        output_pattern = out_folder / f"{vid_id}_%05d.jpg"

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(vid_path.resolve()),
            "-safe",
            "0",
            "-vf",
            f"fps={fps},scale=512:-1:flags=bicubic",
            "-vsync",
            "2",
            "-qscale:v",
            "2",
            str(output_pattern),
        ]

        if debug:
            print("🔍 FFmpeg command:", " ".join(ffmpeg_cmd))

        try:
            subprocess.run(
                ffmpeg_cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            log = e.stderr.decode("utf-8", errors="ignore") if e.stderr else ""
            print(f"\n❌ Frame extraction failed: {vid_path}")
            if debug:
                print(log[:400])
            failed.append(str(vid_path))

    print("\n🎉 Frame extraction finished.")
    if failed:
        fail_log = output_path / "failed_videos.txt"
        fail_log.write_text("\n".join(failed), encoding="utf-8")
        print(f"⚠️ {len(failed)} videos failed; see {fail_log}")


def generate_clip_txt(video_frames_dir: Path, txt_path: Path) -> int:
    """
    Write one line per frame path (sorted by filename) for a single case directory.
    Returns the number of frames written.
    """
    frame_files = sorted(
        (p for p in video_frames_dir.iterdir() if p.is_file()),
        key=lambda p: p.name,
    )
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with txt_path.open("w", encoding="utf-8") as f:
        for frame_path in frame_files:
            f.write(str(frame_path).replace("\\", "/") + "\n")
    return len(frame_files)


def _read_label_table(label_path: Path) -> pd.DataFrame:
    """Load ``Frame`` / ``Phase`` from a label txt (whitespace-separated)."""
    df = pd.read_csv(label_path, sep=r"\s+", engine="python")
    if "Frame" not in df.columns or "Phase" not in df.columns:
        raise ValueError(f"Expected columns Frame, Phase in {label_path}, got {list(df.columns)}")
    return df


def _majority_phase_for_case(
    case_id: int,
    frames_dir: Path,
    label_df: pd.DataFrame,
    debug: bool = False,
) -> Tuple[Optional[int], int]:
    """
    Map original ``Phase`` (1–7) to 0–6; collect phases only when the frame file exists.
    Returns ``(majority_phase_0_6_or_None, num_valid_frames)``.
    """
    case_name = f"{case_id:02d}"
    phases: List[int] = []

    for _, row in label_df.iterrows():
        frame_num = int(row["Frame"])
        phase_raw = int(row["Phase"])
        phase_0_6 = phase_raw - 1
        if phase_0_6 not in PHASE_ID_TO_NAME:
            if debug:
                print(f"⚠️ Case {case_id}: skip unknown Phase {phase_raw} in row Frame={frame_num}")
            continue

        frame_name = f"{case_name}_{frame_num:08d}.jpg"
        frame_path = frames_dir / frame_name
        if not frame_path.is_file():
            if debug:
                print(f"⚠️ Case {case_id}: labeled frame missing on disk: {frame_path}")
            continue

        phases.append(phase_0_6)

    if not phases:
        return None, 0

    counts = Counter(phases)
    top = counts.most_common()
    best_count = top[0][1]
    candidates = sorted(pid for pid, c in top if c == best_count)
    return candidates[0], len(phases)


def build_metadata(
    frames_root: Path,
    labels_root: Path,
    output_dir: Path,
    debug: bool = False,
) -> Dict[str, List[dict]]:
    """
    For each case under the fixed splits, read labels, write ``clip_infos/XX.txt``, and build
    metadata dicts keyed by ``train`` / ``val`` / ``test``.
    """
    clip_infos_dir = output_dir / "clip_infos"
    clip_infos_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(labels_root.glob("label_*.txt"))
    if not label_files:
        print(f"⚠️ No label_*.txt files under {labels_root}")

    by_split: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}

    for label_path in tqdm(label_files, desc="Building metadata"):
        stem = label_path.stem  # e.g. label_01
        try:
            case_id = int(stem.split("_")[-1])
        except ValueError:
            print(f"⚠️ Cannot parse case id from {label_path.name}, skip.")
            continue

        split_name = split_for_case(case_id)
        if split_name is None:
            if debug:
                print(f"⚠️ Case {case_id} not in train/val/test ranges, skip.")
            continue

        case_folder_name = f"{case_id:02d}"
        frames_dir = frames_root / case_folder_name
        if not frames_dir.is_dir():
            print(f"⚠️ Frames directory missing: {frames_dir}, skip case {case_id}.")
            continue

        try:
            label_df = _read_label_table(label_path)
        except Exception as e:
            print(f"⚠️ Failed to read {label_path}: {e}")
            continue

        txt_path = clip_infos_dir / f"{case_folder_name}.txt"
        n_frames = generate_clip_txt(frames_dir, txt_path)
        if n_frames == 0:
            print(f"⚠️ No frames in {frames_dir}, skip case {case_id}.")
            continue

        majority_phase, n_labeled = _majority_phase_for_case(
            case_id, frames_dir, label_df, debug=debug
        )
        if majority_phase is None:
            print(f"⚠️ No valid labeled frames on disk for case {case_id}, skip metadata row.")
            continue

        item = {
            "Index": 0,
            "clip_path": str(txt_path).replace("\\", "/"),
            "label": int(majority_phase),
            "label_name": PHASE_ID_TO_NAME[majority_phase],
            "case_id": int(case_id),
            "clip_idx": 0,
        }
        if debug:
            print(
                f"🔍 Case {case_id}: frames in txt={n_frames}, "
                f"labeled frames matched for majority={n_labeled}, label={majority_phase}"
            )

        by_split[split_name].append(item)

    # Re-index each split to 0..N-1 for cleaner per-split CSVs
    for sname, rows in by_split.items():
        for i, row in enumerate(rows):
            row["Index"] = i

    return by_split


def save_split_csv(output_dir: Path, split_name: str, metadata: List[dict]) -> None:
    """Write ``<split>_metadata.csv`` under ``output_dir``."""
    path = output_dir / f"{split_name}_metadata.csv"
    if not metadata:
        print(f"⚠️ No rows for split '{split_name}', skip writing {path.name}.")
        return
    df = pd.DataFrame(metadata)
    df.to_csv(path, index=False)
    print(f"💾 Saved {len(metadata)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AutoLaparo: clip txts + train/val/test metadata CSVs (end-to-end pipeline style)."
    )
    parser.add_argument(
        "--frames_root",
        type=str,
        default="data/Surge_Frames/AutoLaparo/frames",
        help="Root directory of per-case frame folders (XX/XX_########.jpg).",
    )
    parser.add_argument(
        "--labels_root",
        type=str,
        default="data/Landscopy/autolaparo/task1/labels",
        help="Directory containing label_XX.txt (Frame, Phase columns).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/Surge_Frames/AutoLaparo",
        help="Output directory for clip_infos/ and *_metadata.csv files.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=1,
        help="FPS passed to videos_to_frames() when you enable extraction (default 1).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Verbose warnings (missing frames, ffmpeg, unknown phases).",
    )
    args = parser.parse_args()

    frames_root = Path(args.frames_root)
    labels_root = Path(args.labels_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Frames are usually pre-extracted for AutoLaparo; uncomment if extracting from videos:
    # videos_to_frames(Path("path/to/videos"), frames_root, fps=args.fps, debug=args.debug)

    by_split = build_metadata(
        frames_root=frames_root,
        labels_root=labels_root,
        output_dir=output_dir,
        debug=args.debug,
    )

    for split_name in ("train", "val", "test"):
        save_split_csv(output_dir, split_name, by_split[split_name])

    print("🎉 AutoLaparo preprocessing done.")


if __name__ == "__main__":
    main()
