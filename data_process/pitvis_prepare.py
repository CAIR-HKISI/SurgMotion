#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PitVis (neurosurgery phase recognition) preprocessing — end-to-end pipeline style.

Expected layout:
  - Annotations per video: ``<annot_dir>/annotations_XX.csv`` (XX = 2-digit video id).
    Columns: int_video, int_time, int_step, int_instrument1, int_instrument2
  - Frames at 1 fps: ``<frames_root>/video_XX/video_XX_XXXXXXXX.jpg``
    Frame index (1-based): ``int_time + 1``

This script:
1. Optionally extracts frames via ``videos_to_frames()`` (stub / parity with other datasets;
   frames are usually pre-extracted).
2. For each video in the defined splits, writes ``<output_dir>/clip_infos/video_XX.txt``
   listing all frame paths (one per line, sorted by filename).
3. Reads annotation CSVs, drops phases ``-1, 11, 13``, remaps remaining steps to
   contiguous labels ``0..11``, and emits **one metadata row per annotated frame** that
   survives filtering and exists on disk.
4. Saves ``train_metadata.csv``, ``val_metadata.csv``, ``test_metadata.csv`` under
   ``output_dir``.

Output columns:
  Index, clip_path, label, label_name, case_id, clip_idx

  - clip_path: path to that video’s ``clip_infos/video_XX.txt`` (same for all frames of the video).
  - label / label_name: remapped phase id and original phase name string.
  - case_id: video id string (e.g. ``01``).
  - clip_idx: always ``0``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
from tqdm import tqdm

# Original int_step -> human-readable phase name
PHASE_MAPPING: Dict[int, str] = {
    -1: "operation_ended",
    1: "nasal corridor creation",
    2: "anterior sphenoidotomy",
    3: "septum displacement",
    4: "sphenoid sinus clearance",
    5: "sellotomy",
    6: "durotomy",
    7: "tumour excision",
    8: "haemostasis",
    9: "synthetic_graft_placement",
    10: "fat graft placement",
    11: "gasket seal construct",
    12: "dural sealant",
    13: "nasal packing",
    14: "debris clearance",
}

# Phases excluded from training metadata
FILTERED_PHASES: Set[int] = {-1, 11, 13}

# Sorted valid originals [1,2,...,10,12,14] -> 0..11
VALID_PHASES: List[int] = sorted(p for p in PHASE_MAPPING if p not in FILTERED_PHASES)
PHASE_REMAP: Dict[int, int] = {p: i for i, p in enumerate(VALID_PHASES)}

TRAIN_VIDEOS: List[str] = [
    "01",
    "03",
    "04",
    "05",
    "07",
    "08",
    "09",
    "10",
    "11",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "25",
]
VAL_VIDEOS: List[str] = ["02", "06", "12", "13", "24"]
TEST_VIDEOS: List[str] = ["02", "06", "12", "13", "24"]

EXPECTED_ANNOT_COLS = [
    "int_video",
    "int_time",
    "int_step",
    "int_instrument1",
    "int_instrument2",
]


def splits_for_video(video_id: str) -> List[str]:
    """Return which splits (train / val / test) contain this video id."""
    out: List[str] = []
    if video_id in TRAIN_VIDEOS:
        out.append("train")
    if video_id in VAL_VIDEOS:
        out.append("val")
    if video_id in TEST_VIDEOS:
        out.append("test")
    return out


def frame_path_for_row(frames_root: Path, video_id: str, frame_index: int) -> Path:
    """1-based frame index -> expected jpg path under PitVis layout."""
    return frames_root / f"video_{video_id}" / f"video_{video_id}_{frame_index:08d}.jpg"


def videos_to_frames(
    input_path: Path,
    output_path: Path,
    fps: int = 1,
    debug: bool = False,
) -> None:
    """
    Stub for pipeline parity: PitVis frames are normally already extracted at 1 fps.

    Implement ffmpeg-based extraction here if you obtain raw videos and want the same
    interface as other ``*_prepare.py`` scripts.
    """
    if debug:
        print(
            "videos_to_frames: stub (no extraction). "
            f"input_path={input_path}, output_path={output_path}, fps={fps}"
        )


def generate_clip_txt(video_frames_dir: Path, txt_path: Path) -> int:
    """
    Write one line per frame path (sorted by filename) for a single video folder.
    Returns the number of lines written.
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


def build_metadata(
    frames_root: Path,
    annot_dir: Path,
    output_dir: Path,
    debug: bool = False,
) -> Dict[str, List[dict]]:
    """
    Read annotation CSVs, filter phases, write clip txts, and build per-frame metadata
    rows grouped by split name.
    """
    clip_infos_dir = output_dir / "clip_infos"
    clip_infos_dir.mkdir(parents=True, exist_ok=True)

    by_split: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}

    all_videos = sorted(set(TRAIN_VIDEOS) | set(VAL_VIDEOS) | set(TEST_VIDEOS))

    for video_id in tqdm(all_videos, desc="PitVis videos"):
        splits = splits_for_video(video_id)
        if not splits:
            continue

        frames_dir = frames_root / f"video_{video_id}"
        if not frames_dir.is_dir():
            print(f"⚠️ Missing frames directory: {frames_dir}, skip video {video_id}.")
            continue

        txt_path = clip_infos_dir / f"video_{video_id}.txt"
        n_written = generate_clip_txt(frames_dir, txt_path)
        if n_written == 0:
            print(f"⚠️ No frames under {frames_dir}, skip video {video_id}.")
            continue

        annot_path = annot_dir / f"annotations_{video_id}.csv"
        if not annot_path.is_file():
            print(f"⚠️ Missing annotation file: {annot_path}, skip video {video_id}.")
            continue

        try:
            df = pd.read_csv(annot_path)
        except Exception as e:
            print(f"⚠️ Failed to read {annot_path}: {e}")
            continue

        missing = [c for c in EXPECTED_ANNOT_COLS if c not in df.columns]
        if missing:
            print(f"⚠️ {annot_path.name} missing columns {missing}, skip.")
            continue

        clip_rel = str(txt_path).replace("\\", "/")

        for _, row in df.iterrows():
            int_step = int(row["int_step"])
            if int_step in FILTERED_PHASES:
                continue
            if int_step not in PHASE_REMAP:
                if debug:
                    print(f"⚠️ Video {video_id}: unknown int_step={int_step}, skip row.")
                continue

            int_time = int(row["int_time"])
            frame_index = int_time + 1
            fp = frame_path_for_row(frames_root, video_id, frame_index)
            if not fp.is_file():
                if debug:
                    print(f"⚠️ Video {video_id}: frame missing {fp}")
                continue

            label = PHASE_REMAP[int_step]
            label_name = PHASE_MAPPING.get(int_step, f"unknown_{int_step}")

            item = {
                "Index": 0,
                "clip_path": clip_rel,
                "label": label,
                "label_name": label_name,
                "case_id": video_id,
                "clip_idx": 0,
            }

            for s in splits:
                by_split[s].append(dict(item))

    for _split_name, rows in by_split.items():
        for i, row in enumerate(rows):
            row["Index"] = i

    return by_split


def save_split_csv(output_dir: Path, split_name: str, rows: List[dict]) -> None:
    path = output_dir / f"{split_name}_metadata.csv"
    if not rows:
        print(f"⚠️ No rows for split '{split_name}', skip writing {path.name}.")
        return
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved {len(rows)} rows -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PitVis: clip txts + per-frame train/val/test metadata CSVs."
    )
    parser.add_argument(
        "--frames_root",
        type=str,
        default="data/Surge_Frames/PitVis/frames",
        help="Root containing video_XX/ folders with jpg frames.",
    )
    parser.add_argument(
        "--annot_dir",
        type=str,
        default="data/NeuroSurgery/pitvits/26531686",
        help="Directory with annotations_XX.csv per video.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/Surge_Frames/PitVis",
        help="Output directory for clip_infos/ and *_metadata.csv.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Verbose messages (missing frames, unknown phases, stub logging).",
    )
    args = parser.parse_args()

    frames_root = Path(args.frames_root)
    annot_dir = Path(args.annot_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Uncomment when raw videos are available:
    # videos_to_frames(Path("path/to/videos"), frames_root, fps=1, debug=args.debug)

    by_split = build_metadata(
        frames_root=frames_root,
        annot_dir=annot_dir,
        output_dir=output_dir,
        debug=args.debug,
    )

    for split_name in ("train", "val", "test"):
        save_split_csv(output_dir, split_name, by_split[split_name])

    print("PitVis preprocessing done.")


if __name__ == "__main__":
    main()
