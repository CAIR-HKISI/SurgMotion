#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PmLR50 (liver resection phase recognition) preprocessing
--------------------------------------------------------
Expected layout:
  - Frames: ``<frames_root>/XX/XXXXXXXX.jpg`` (``XX`` = zero-padded 2-digit case id).
  - Labels (pickle): each file is ``dict[video_id_str, list[{"frame_id": int, "phase_gt": int}, ...]]``.

This script:
1. Optionally extracts frames via ``videos_to_frames()`` (stub by default; frames are usually shipped).
2. For each case appearing in a split pickle, writes ``<output_dir>/clip_infos/XX.txt`` (all frame paths, sorted).
3. Builds **per-frame** metadata: one row per labeled frame that exists on disk.

Output CSV columns:
  ``Index``, ``clip_path``, ``label``, ``label_name``, ``case_id``, ``clip_idx``

  - ``clip_path``: path to that case’s ``clip_infos/XX.txt`` (same for all frames of the case).
  - ``label`` / ``label_name``: ``phase_gt`` and mapped phase name.
  - ``case_id``: ``int(video_id)``.
  - ``clip_idx``: always ``0`` (one clip list per case).

Writes ``train_metadata.csv``, ``val_metadata.csv``, ``test_metadata.csv``, ``metadata.csv`` (combined),
and ``missing_frames_report.csv`` under ``output_dir``.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

# Phase id → human-readable name (official PmLR50 mapping)
PHASE_ID_TO_NAME: Dict[int, str] = {
    0: "Preparation stage",
    1: "Knotting of the Foley catheter",
    2: "Procedure of the liver resection",
    3: "Release of the Foley catheter",
    4: "Postprocessing stage",
}


def videos_to_frames(
    input_path: Path,
    output_path: Path,
    fps: int = 1,
    debug: bool = False,
) -> None:
    """
    Stub: extract frames from raw videos into PmLR50-style folders ``output_path/XX/XXXXXXXX.jpg``.

    The public release is normally distributed with 1 fps frames already; implement ffmpeg here if you
    have raw videos and need extraction.
    """
    print(
        "videos_to_frames(): stub — PmLR50 frames are usually pre-extracted. "
        "Implement ffmpeg extraction here if needed."
    )
    if debug:
        print(f"  input_path={input_path}, output_path={output_path}, fps={fps}")


def generate_clip_txt(video_frames_dir: Path, txt_path: Path) -> int:
    """
    Write one line per frame path (sorted by filename) for a single case directory.
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


def _load_pickle_labels(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def build_metadata(
    frames_root: Path,
    train_label: Path,
    val_label: Path,
    test_label: Path,
    output_dir: Path,
    debug: bool = False,
) -> Tuple[Dict[str, List[dict]], List[dict]]:
    """
    Load train / val / test pickles, generate clip txts per case, and build per-split per-frame rows.

    Returns ``(rows_by_split, missing_frames)`` where ``rows_by_split`` keys are
    ``"train"``, ``"val"``, ``"test"``.
    """
    clip_infos_dir = output_dir / "clip_infos"
    clip_infos_dir.mkdir(parents=True, exist_ok=True)

    rows_by_split: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    missing_frames: List[dict] = []
    global_index = 0

    split_specs: List[Tuple[str, Path]] = [
        ("train", train_label),
        ("val", val_label),
        ("test", test_label),
    ]

    for split_name, label_path in split_specs:
        if not label_path.is_file():
            print(f"⚠️ Label file not found, skipping split '{split_name}': {label_path}")
            continue

        try:
            data = _load_pickle_labels(label_path)
        except Exception as exc:
            print(f"⚠️ Failed to load {label_path}: {exc}")
            continue

        if not isinstance(data, dict):
            print(f"⚠️ Expected dict in {label_path}, got {type(data)}")
            continue

        for vid, entries in tqdm(
            data.items(),
            desc=f"PmLR50 {split_name}",
        ):
            try:
                case_id = int(vid)
            except (TypeError, ValueError):
                print(f"⚠️ Invalid video id {vid!r} in {label_path.name}, skip.")
                continue

            video_dir = frames_root / f"{case_id:02d}"
            if not video_dir.is_dir():
                print(f"⚠️ Missing frame directory: {video_dir}, skip video {vid!r}.")
                continue

            txt_path = clip_infos_dir / f"{case_id:02d}.txt"
            n_lines = generate_clip_txt(video_dir, txt_path)
            if n_lines == 0:
                print(f"⚠️ No frames listed under {video_dir}, skip case {case_id}.")
                continue

            clip_path_str = str(txt_path).replace("\\", "/")

            if not isinstance(entries, list):
                if debug:
                    print(f"⚠️ Case {case_id}: entries not a list, skip.")
                continue

            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                frame_id = int(entry["frame_id"])
                phase_gt = int(entry.get("phase_gt", -1))
                frame_file = f"{frame_id:08d}.jpg"
                frame_path = video_dir / frame_file

                if not frame_path.is_file():
                    missing_frames.append(
                        {
                            "split": split_name,
                            "case_id": case_id,
                            "missing_frame": frame_file,
                            "label_file": label_path.name,
                        }
                    )
                    continue

                label_name = PHASE_ID_TO_NAME.get(phase_gt, "Unknown")

                row = {
                    "Index": global_index,
                    "clip_path": clip_path_str,
                    "label": phase_gt,
                    "label_name": label_name,
                    "case_id": case_id,
                    "clip_idx": 0,
                }
                global_index += 1
                rows_by_split[split_name].append(row)

    return rows_by_split, missing_frames


def save_split_csv(output_dir: Path, split_name: str, rows: List[dict]) -> None:
    path = output_dir / f"{split_name}_metadata.csv"
    if not rows:
        print(f"⚠️ No rows for split '{split_name}', skip writing {path.name}.")
        return
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"💾 Saved {len(rows)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PmLR50: clip txts + per-frame phase metadata (train/val/test pickles).",
    )
    parser.add_argument(
        "--frames_root",
        type=str,
        default="data/Surge_Frames/PmLR50/frames",
        help="Root with per-case folders XX/XXXXXXXX.jpg.",
    )
    parser.add_argument(
        "--train_label",
        type=str,
        default="data/Landscopy/PmLR50/PmLR50/labels/train/1fpstrain.pickle",
        help="Train pickle (video_id -> list of frame annotations).",
    )
    parser.add_argument(
        "--test_label",
        type=str,
        default="data/Landscopy/PmLR50/PmLR50/labels/infer/1fpsinfer.pickle",
        help="Test pickle (infer split).",
    )
    parser.add_argument(
        "--val_label",
        type=str,
        default="data/Landscopy/PmLR50/PmLR50/labels/test/1fpstest.pickle",
        help="Validation pickle (named test/ on disk).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/Surge_Frames/PmLR50",
        help="Output directory for clip_infos/ and CSVs.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Extra diagnostics for malformed annotations.",
    )
    args = parser.parse_args()

    frames_root = Path(args.frames_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Uncomment when extracting from raw videos:
    # videos_to_frames(Path("path/to/videos"), frames_root, fps=1, debug=args.debug)

    rows_by_split, missing_frames = build_metadata(
        frames_root=frames_root,
        train_label=Path(args.train_label),
        val_label=Path(args.val_label),
        test_label=Path(args.test_label),
        output_dir=output_dir,
        debug=args.debug,
    )

    for split_name in ("train", "val", "test"):
        save_split_csv(output_dir, split_name, rows_by_split[split_name])

    all_rows: List[dict] = (
        rows_by_split["train"] + rows_by_split["val"] + rows_by_split["test"]
    )
    combined_path = output_dir / "metadata.csv"
    if all_rows:
        pd.DataFrame(all_rows).to_csv(combined_path, index=False)
        print(f"💾 Saved combined metadata ({len(all_rows)} rows) to {combined_path}")
    else:
        print("⚠️ No metadata rows; skip writing metadata.csv")

    miss_path = output_dir / "missing_frames_report.csv"
    miss_cols = ["split", "case_id", "missing_frame", "label_file"]
    if missing_frames:
        pd.DataFrame(missing_frames).to_csv(miss_path, index=False)
        print(f"⚠️ Missing frames: {len(missing_frames)} entries → {miss_path}")
    else:
        pd.DataFrame(columns=miss_cols).to_csv(miss_path, index=False)
        print(f"✅ No missing frames; wrote empty report: {miss_path}")

    print("🎉 PmLR50 preprocessing done.")


if __name__ == "__main__":
    main()
