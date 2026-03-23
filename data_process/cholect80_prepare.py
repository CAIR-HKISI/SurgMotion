#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cholec80 preprocessing script
--------------------------------
Dataset layout (expected on disk):
  - Phase annotations at 25 fps (tab-separated ``Frame<TAB>Phase``):
      ``<annot_dir>/videoXX-phase.txt`` for ``video01`` … ``video80``.
  - Frames extracted at 1 fps (one folder per video):
      ``<frames_root>/videoXX/*.jpg``

Pipeline steps performed by this script:
  1. (Optional) ``videos_to_frames`` — extract 1 fps JPEGs with ffmpeg; keep
     commented in ``main`` if frames are already extracted.
  2. ``generate_clip_txt`` — for each case, write ``clip_infos/videoXX.txt``
     listing every frame path (one path per line, forward slashes).
  3. ``build_metadata`` — align each extracted frame with annotation rows using
     ``step = fps_orig // fps_target`` (default 25 // 1 = 25), i.e. use every
     25-th annotation row for successive 1 fps frames.
  4. Save split CSVs under ``output_dir``:
      ``train_metadata.csv`` (videos 1–40), ``test_metadata.csv`` (41–80).

Output CSV columns:
  - ``Index``: row index within that split CSV (0-based).
  - ``clip_path``: path to the case clip list txt under ``output_dir/clip_infos/``.
  - ``label``: integer phase id (0–6); unknown phase strings map to ``-1``.
  - ``label_name``: phase name string from the annotation row used for that frame.
  - ``case_id``: numeric video id (1–80), parsed from ``videoXX``.
  - ``clip_idx``: always ``0`` (single clip txt per case covering all frames).
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

# Original annotation sampling rate and default target frame rate (must divide 25).
FPS_ORIG = 25

# Phase name -> class id (Cholec80 seven phases).
PHASE_TO_ID: Dict[str, int] = {
    "Preparation": 0,
    "CalotTriangleDissection": 1,
    "ClippingCutting": 2,
    "GallbladderDissection": 3,
    "GallbladderPackaging": 4,
    "CleaningCoagulation": 5,
    "GallbladderRetraction": 6,
}


def _split_for_case(case_num: int) -> Optional[str]:
    """Return ``train`` (1–40), ``test`` (41–80), or ``None`` if out of range."""
    if 1 <= case_num <= 40:
        return "train"
    if 41 <= case_num <= 80:
        return "test"
    return None


def _parse_phase_annotation_path(path: Path) -> Optional[Tuple[str, int]]:
    """
    Parse ``videoXX-phase.txt`` into ``(case_folder_name, case_num)``.
    Returns ``None`` if the filename does not match the expected pattern.
    """
    name = path.name
    if not name.endswith("-phase.txt"):
        return None
    stem = name[: -len("-phase.txt")]
    if not stem.startswith("video"):
        return None
    try:
        case_num = int(stem[len("video") :])
    except ValueError:
        return None
    return stem, case_num


def videos_to_frames(
    input_path: Path,
    output_path: Path,
    fps: int = 1,
    debug: bool = False,
) -> None:
    """
    Extract frames from all ``*.mp4`` under ``input_path`` into
    ``output_path/videoXX/videoXX_%05d.jpg`` (same layout as typical Cholec80
    frame folders). Intended to be called from ``main`` only when raw videos
    are available; left commented when frames already exist.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    video_files = sorted(input_path.glob("*.mp4"))

    if not video_files:
        print(f"⚠️ No mp4 videos found under {input_path}.")
        return

    print(f"\n🎞️ Found {len(video_files)} videos, extracting frames at fps={fps}...\n")
    failed: List[str] = []

    for vid_path in tqdm(video_files, desc="Extracting frames"):
        vid_id = vid_path.stem
        out_folder = output_path / vid_id
        out_folder.mkdir(parents=True, exist_ok=True)
        pattern = out_folder / f"{vid_id}_%05d.jpg"
        cmd = [
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
            str(pattern),
        ]
        if debug:
            print("🔍 FFmpeg:", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode("utf-8", errors="ignore") if e.stderr else ""
            print(f"\n❌ Failed: {vid_path}")
            if debug:
                print(err[:400])
            failed.append(str(vid_path))

    print("\n🎉 Frame extraction finished.")
    if failed:
        log = output_path / "failed_videos.txt"
        log.write_text("\n".join(failed), encoding="utf-8")
        print(f"⚠️ {len(failed)} videos failed; see {log}")


def _sorted_jpeg_paths(video_frames_dir: Path) -> List[Path]:
    """Return sorted ``.jpg`` / ``.jpeg`` paths under a case frame directory."""
    return sorted(
        (
            p
            for p in video_frames_dir.iterdir()
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg")
        ),
        key=lambda p: p.name,
    )


def generate_clip_txt(video_frames_dir: Path, txt_path: Path) -> int:
    """
    Write all JPEG frame paths under ``video_frames_dir`` to ``txt_path``
    (one path per line, POSIX-style slashes). Returns the number of frames.
    """
    frame_files = _sorted_jpeg_paths(video_frames_dir)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with txt_path.open("w", encoding="utf-8") as f:
        for frame_path in frame_files:
            f.write(str(frame_path).replace("\\", "/") + "\n")
    return len(frame_files)


def build_metadata(
    frames_root: Path,
    annot_dir: Path,
    output_dir: Path,
    fps_target: int = 1,
    debug: bool = False,
) -> Tuple[List[dict], List[dict]]:
    """
    For each Cholec80 case:
      - Read 25 fps phase annotations.
      - Emit ``clip_infos/<case_name>.txt`` listing all frames in sorted order.
      - Add one metadata row per extracted frame (phase label from annotation
        row index ``min(frame_index * step, n_rows - 1)``).

    Returns ``(train_rows, test_rows)`` with fields required by the project CSV
    schema. ``case_id`` is the numeric video id; ``clip_idx`` is always ``0``.
    """
    if FPS_ORIG % fps_target != 0:
        raise ValueError(
            f"fps_target={fps_target} must divide FPS_ORIG={FPS_ORIG} "
            "so annotation subsampling stays aligned."
        )
    step = FPS_ORIG // fps_target

    clip_infos_dir = output_dir / "clip_infos"
    clip_infos_dir.mkdir(parents=True, exist_ok=True)

    annot_paths = sorted(annot_dir.glob("video*-phase.txt"))
    train_meta: List[dict] = []
    test_meta: List[dict] = []
    train_index = 0
    test_index = 0

    for annot_path in tqdm(annot_paths, desc="Cholec80 cases"):
        parsed = _parse_phase_annotation_path(annot_path)
        if parsed is None:
            if debug:
                print(f"⚠️ Skip unexpected annotation file: {annot_path.name}")
            continue
        case_name, case_num = parsed
        split = _split_for_case(case_num)
        if split is None:
            if debug:
                print(f"⚠️ Skip {case_name}: case id outside train/test range.")
            continue

        df = pd.read_csv(annot_path, sep="\t")
        df.columns = [str(c).strip() for c in df.columns]
        if "Phase" not in df.columns:
            raise KeyError(f"'Phase' column missing in {annot_path}")
        n_ann = len(df)
        if n_ann == 0:
            print(f"⚠️ Empty annotations: {annot_path}")
            continue

        video_frames_dir = frames_root / case_name
        if not video_frames_dir.is_dir():
            print(f"⚠️ Missing frame directory: {video_frames_dir}")
            continue

        clip_txt = clip_infos_dir / f"{case_name}.txt"
        n_frames = generate_clip_txt(video_frames_dir, clip_txt)
        if n_frames == 0:
            print(f"⚠️ No frames in {video_frames_dir}")
            continue

        clip_path_str = str(clip_txt).replace("\\", "/")
        target_list = train_meta if split == "train" else test_meta

        frame_files = _sorted_jpeg_paths(video_frames_dir)

        for frame_i, _frame_path in enumerate(frame_files):
            row_idx = min(frame_i * step, n_ann - 1)
            phase_name = str(df.iloc[row_idx]["Phase"]).strip()
            label = PHASE_TO_ID.get(phase_name, -1)

            if split == "train":
                idx = train_index
                train_index += 1
            else:
                idx = test_index
                test_index += 1

            target_list.append(
                {
                    "Index": idx,
                    "clip_path": clip_path_str,
                    "label": int(label),
                    "label_name": phase_name,
                    "case_id": int(case_num),
                    "clip_idx": 0,
                }
            )

    return train_meta, test_meta


def save_metadata_csv(path: Path, rows: List[dict]) -> None:
    """Write metadata rows to CSV (UTF-8, no index column)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"💾 Saved {len(rows)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cholec80: build clip_info txts and train/test metadata CSVs "
            "from 1 fps frames and 25 fps phase annotations."
        )
    )
    parser.add_argument(
        "--frames_root",
        type=str,
        default="data/Surge_Frames/Cholec80/frames",
        help="Root folder containing per-video frame directories (video01, …).",
    )
    parser.add_argument(
        "--annot_dir",
        type=str,
        default="data/Landscopy/cholec80/phase_annotations",
        help="Directory with videoXX-phase.txt tab-separated annotation files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/Surge_Frames/Cholec80",
        help="Output directory for clip_infos/ and *_metadata.csv files.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=1,
        help="Target frame extraction rate used to align with 25 fps labels (default 1).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print extra diagnostics (e.g. ffmpeg commands, skipped files).",
    )
    args = parser.parse_args()

    frames_root = Path(args.frames_root)
    annot_dir = Path(args.annot_dir)
    output_dir = Path(args.output_dir)

    # Uncomment when raw videos are available, and set input_path accordingly.
    # videos_to_frames(
    #     input_path=Path("data/Landscopy/cholec80/videos"),
    #     output_path=frames_root,
    #     fps=args.fps,
    #     debug=args.debug,
    # )

    train_rows, test_rows = build_metadata(
        frames_root=frames_root,
        annot_dir=annot_dir,
        output_dir=output_dir,
        fps_target=args.fps,
        debug=args.debug,
    )

    save_metadata_csv(output_dir / "train_metadata.csv", train_rows)
    save_metadata_csv(output_dir / "test_metadata.csv", test_rows)


if __name__ == "__main__":
    main()
