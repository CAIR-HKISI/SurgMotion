#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OphNet2024 Preprocessing Pipeline
----------------------------------
Aligned with NSJepa_jinlin/data_process/ophnet_csv.py.

Expected layout:
  - Phase CSV: data/Ophthalmology/OphNet2024_trimmed_phase/OphNet2024_loca_challenge_phase.csv
    Columns: video_id, start, end, phase_id, split (train / val / test).
  - Frames (pre-extracted): data/Surge_Frames/OphNet2024_phase/frames/{video_id}_{clip_index}/*.jpg
    Clip index = per-video_id cumulative index after sorting rows by start.

NOTE: This pipeline assumes frames already exist. The OphNet dataset distributes
clips, not raw videos. Use --step frames only if you have the original videos.

Pipeline Steps:
  --step all (default): metadata + clips (does NOT extract frames)
  --step frames:        Extract frames from videos (requires --videos_dir with mp4s)
  --step metadata:      Build frame-level metadata CSV
  --step clips:         Generate dense sliding-window clips

Output structure:
  <output_dir>/
    clip_infos/                    # One txt per clip
    train_metadata.csv             # Frame-level metadata
    val_metadata.csv
    test_metadata.csv
    missing_frames_report.csv
    clips_64f/                     # Dense clips
      train_dense_64f_detailed.csv
      ...

Usage:
    python ophnet_prepare.py --step all
    python ophnet_prepare.py --step metadata
    python ophnet_prepare.py --step frames --videos_dir /path/to/videos
    python ophnet_prepare.py --step clips --window_size 64
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from gen_clips import generate_dense_clips

# 96 phases (0-95)
PHASE2NAME: Dict[int, str] = {
    0: "Viscoelastic Injection",
    1: "Nuclear Management (for cataract surgery)",
    2: "Step Interval",
    3: "Non-functional Segment",
    4: "Intraocular Lens Implantation",
    5: "Incision Closure",
    6: "Corneal Incision Creation",
    7: "Capsulorhexis",
    8: "Cortex Aspiration",
    9: "Recipient Corneal Bed Preparation",
    10: "Anterior Chamber Injection/Washing",
    11: "Viscoelastic Aspiration",
    12: "Capsular Membrane Staining",
    13: "Corneal Graft Suturing",
    14: "Viscoelastic Application on Cornea",
    15: "Corneal-Scleral Tunnel Creation",
    16: "Swab Wiping",
    17: "Conjunctival Incision Creation",
    18: "Anterior Chamber Gas Injection",
    19: "Ocular Surface Irrigation",
    20: "Surgical Marking",
    21: "Subconjunctival Drug Injection",
    22: "Donor Corneal Graft Preparation",
    23: "Scleral Hemostasis",
    24: "Corneal Interlamellar Injection",
    25: "Drainage Tube Implantation",
    26: "Use of Iris Expander",
    27: "Drainage Device Preparation",
    28: "Scleral Flap Creation",
    29: "Anterior Chamber Washing",
    30: "Suspension Suture",
    31: "Drainage Device Implantation",
    32: "Intraoperative Gonioscopy Application",
    33: "Corneal Measurement",
    34: "Scleral Flap Suturing",
    35: "Scleral Tunnel Creation",
    36: "Placement of Bandage Contact Lens",
    37: "Peripheral Iridectomy",
    38: "Anterior Chamber Drainage Device Implantation",
    39: "Antimetabolite Application",
    40: "Scleral Support Ring Manipulation",
    41: "Deep Sclerectomy",
    42: "Anterior Vitrectomy",
    43: "Placement of Eyelid Speculum",
    44: "Hooking of Extraocular Muscle",
    45: "Observation of Corneal Astigmatism",
    46: "Pupil Dilation",
    47: "Iris Prolapse Management",
    48: "Measurement on the Scleral",
    49: "Goniotomy",
    50: "Trabeculectomy",
    51: "Instrument Fabrication",
    52: "Capsular Tension Ring Implantation",
    53: "Femtosecond Laser-Assisted Corneal Transplantation",
    54: "Removal of Pupillary/Iris Fibrosis Membrane",
    55: "Allograft/Biological Tissue Trimming",
    56: "Corneal Suture Removal",
    57: "Iris Synechiae Separation",
    58: "Placement of Sponge on Cornea",
    59: "Femtosecond Laser-Assisted Cataract Surgery",
    60: "Astigmatism Axis Gauge",
    61: "Microcatheter Insertion into Trabecular Meshwork",
    62: "Sub-Iris Exploration",
    63: "Artificial Cornea Manipulaiton",
    64: "Scleral Puncture/Incision",
    65: "Astigmatic Keratotomy",
    66: "Scleral Fixation of Intraocular Lens",
    67: "Removal of Fascia Tissue",
    68: "Canthotomy",
    69: "Removal of Lens Fibrotic Membrane",
    70: "Amniotic Membrane Transplantation",
    71: "Pupilloplasty",
    72: "Anterior Chamber Irrigation",
    73: "Sub-Tenon Injection",
    74: "Sclerectomy",
    75: "Corneal Interlamellar Irrigation",
    76: "Cyclophotocoagulation",
    77: "Drainage Device Adjustment",
    78: "Schlemm's Canal Inner Wall Removal",
    79: "Scleral Flap Incision Inspection",
    80: "Lens Extraction",
    81: "Anterior Chamber Vitreous Cleaning",
    82: "Intraocular Lens Removal",
    83: "Pterygium Excision",
    84: "Scleral Suture",
    85: "Drainage Tube Removal",
    86: "Suture Trimming",
    87: "Conjunctival Vessel Examination",
    88: "Special Puncture Knife Traversing the Anterior Chamber",
    89: "Removal of Object from Anterior Chamber",
    90: "Anterior Chamber Inspection",
    91: "Iris Repositioning",
    92: "Suprachoroidal Space Separation",
    93: "Scleral Flap Embedding in the Suprachoroidal Space",
    94: "Vitreoretinal Surgery",
    95: "Conjunctival Trimming",
}


def videos_to_frames(
    input_path: Path,
    output_path: Path,
    fps: int = 30,
    debug: bool = False,
) -> None:
    """
    Extract frames from all *.mp4 under input_path into output_path.
    Output: output_path/<video_stem>/<video_stem>_%08d.jpg
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    video_files = sorted(input_path.glob("*.mp4"))

    if not video_files:
        print(f"[WARN] No mp4 videos found under {input_path}.")
        return

    print(f"\n[INFO] Found {len(video_files)} videos, extracting frames at {fps} fps...")
    failed: List[str] = []

    for vid_path in tqdm(video_files, desc="Extracting frames"):
        vid_id = vid_path.stem
        out_folder = output_path / vid_id
        out_folder.mkdir(parents=True, exist_ok=True)
        pattern = out_folder / f"{vid_id}_%08d.jpg"

        cmd = [
            "ffmpeg", "-y", "-i", str(vid_path.resolve()),
            "-vf", f"fps={fps},scale=512:-1:flags=bicubic",
            "-vsync", "2", "-qscale:v", "2",
            str(pattern),
        ]

        if debug:
            print(f"[DEBUG] FFmpeg: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed: {vid_path}")
            if debug and e.stderr:
                print(e.stderr.decode("utf-8", errors="ignore")[:500])
            failed.append(str(vid_path))

    print(f"[INFO] Frame extraction finished.")
    if failed:
        log = output_path / "failed_videos.txt"
        log.write_text("\n".join(failed), encoding="utf-8")
        print(f"[WARN] {len(failed)} videos failed; see {log}")


def _list_jpeg_frames(video_frames_dir: Path) -> List[Path]:
    """Sorted .jpg/.jpeg files under a clip directory."""
    return sorted(
        (p for p in video_frames_dir.iterdir()
         if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg")),
        key=lambda p: p.name,
    )


def generate_clip_txt(video_frames_dir: Path, txt_path: Path) -> List[str]:
    """
    Write one line per JPEG frame under video_frames_dir.
    Returns list of frame paths.
    """
    if not video_frames_dir.is_dir():
        return []

    frame_files = _list_jpeg_frames(video_frames_dir)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    frame_paths = [str(fp).replace("\\", "/") for fp in frame_files]
    with txt_path.open("w", encoding="utf-8") as f:
        for fp in frame_paths:
            f.write(fp + "\n")
    return frame_paths


def _parse_case_id(video_id: str) -> int:
    """Strip optional case_ prefix and parse integer case id."""
    vid = str(video_id).strip()
    if vid.startswith("case_"):
        vid = vid[5:]
    return int(vid)


def build_frame_level_metadata(
    frames_root: Path,
    label_csv: Path,
    output_dir: Path,
    debug: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Build frame-level metadata with columns: Case_ID, Frame_Path, Phase_GT, Phase_Name.
    Returns dict of DataFrames keyed by split name.
    """
    if not label_csv.is_file():
        raise FileNotFoundError(f"Label CSV not found: {label_csv}")

    df = pd.read_csv(label_csv)
    required = {"video_id", "start", "phase_id", "split"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Label CSV missing columns {missing_cols}")

    df = df.sort_values(["video_id", "start"]).reset_index(drop=True)
    df["clip_index"] = df.groupby("video_id", sort=False).cumcount()

    clip_infos_dir = output_dir / "clip_infos"
    clip_infos_dir.mkdir(parents=True, exist_ok=True)

    by_split: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    missing_report: List[dict] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building frame-level metadata"):
        video_id = str(row["video_id"]).strip()
        clip_index = int(row["clip_index"])
        split_raw = str(row["split"]).strip().lower()
        phase_id = int(row["phase_id"])
        label_name = PHASE2NAME.get(phase_id, "Unknown Phase")

        if split_raw not in ("train", "val", "test"):
            missing_report.append({
                "video_id": video_id,
                "clip_idx": clip_index,
                "split": split_raw,
                "reason": "unknown_split",
            })
            continue

        try:
            case_id = _parse_case_id(video_id)
        except ValueError:
            missing_report.append({
                "video_id": video_id,
                "clip_idx": clip_index,
                "split": split_raw,
                "reason": "bad_video_id",
            })
            continue

        clip_key = f"{video_id}_{clip_index}"
        clip_dir = frames_root / clip_key
        txt_path = clip_infos_dir / f"{clip_key}.txt"

        if not clip_dir.is_dir():
            missing_report.append({
                "video_id": video_id,
                "clip_idx": clip_index,
                "split": split_raw,
                "reason": "missing_clip_directory",
            })
            if debug:
                print(f"[DEBUG] Missing clip directory: {clip_dir}")
            continue

        frame_paths = generate_clip_txt(clip_dir, txt_path)
        if not frame_paths:
            missing_report.append({
                "video_id": video_id,
                "clip_idx": clip_index,
                "split": split_raw,
                "reason": "empty_clip_directory",
            })
            continue

        for fp in frame_paths:
            by_split[split_raw].append({
                "Case_ID": case_id,
                "Frame_Path": fp,
                "Phase_GT": phase_id,
                "Phase_Name": label_name,
            })

    if missing_report:
        miss_path = output_dir / "missing_frames_report.csv"
        pd.DataFrame(missing_report).to_csv(miss_path, index=False)
        print(f"[WARN] Missing/skipped clips: {len(missing_report)} entries (see {miss_path})")

    result = {}
    for split_name, rows in by_split.items():
        if rows:
            df = pd.DataFrame(rows)
            df = df.sort_values(["Case_ID", "Frame_Path"]).reset_index(drop=True)
            result[split_name] = df
        else:
            result[split_name] = pd.DataFrame(
                columns=["Case_ID", "Frame_Path", "Phase_GT", "Phase_Name"]
            )

    return result


def save_metadata_csvs(output_dir: Path, metadata_by_split: Dict[str, pd.DataFrame]) -> None:
    """Save frame-level metadata CSVs."""
    for split_name, df in metadata_by_split.items():
        path = output_dir / f"{split_name}_metadata.csv"
        if len(df) == 0:
            print(f"[WARN] No rows for split '{split_name}', skip writing {path.name}.")
            continue
        df.to_csv(path, index=False)
        print(f"[INFO] Saved {len(df)} frame rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OphNet2024: End-to-end preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python ophnet_prepare.py --step all
    python ophnet_prepare.py --step metadata
    python ophnet_prepare.py --step clips --window_size 64
        """,
    )
    parser.add_argument(
        "--step",
        choices=["all", "frames", "metadata", "clips"],
        default="all",
        help="Pipeline step to run (default: all)",
    )
    parser.add_argument(
        "--videos_dir",
        type=str,
        default="",
        help="Directory containing raw mp4 videos (only for --step frames; not part of standard workflow)",
    )
    parser.add_argument(
        "--frames_root",
        type=str,
        default="data/Surge_Frames/OphNet2024_phase/frames",
        help="Root containing clip folders {video_id}_{clip_index}/ with .jpg frames",
    )
    parser.add_argument(
        "--label_csv",
        type=str,
        default="data/Ophthalmology/OphNet2024_trimmed_phase/OphNet2024_loca_challenge_phase.csv",
        help="CSV with video_id, start, end, phase_id, split",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/Surge_Frames/OphNet2024_phase",
        help="Output directory for metadata and clips",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="FPS for frame extraction (default: 30)",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=64,
        help="Window size for dense clip generation (default: 64)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride for dense clip generation (default: 1)",
    )
    parser.add_argument(
        "--no_padding",
        action="store_true",
        help="Disable padding for incomplete windows",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug output",
    )
    args = parser.parse_args()

    frames_root = Path(args.frames_root)
    label_csv = Path(args.label_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("OphNet2024 Preprocessing Pipeline")
    print("=" * 60)

    # Frame extraction is NOT part of --step all; run only with --step frames
    if args.step == "frames":
        print("\n[STEP] Extracting frames from videos...")
        videos_dir = Path(args.videos_dir)
        if videos_dir.exists() and any(videos_dir.glob("*.mp4")):
            videos_to_frames(videos_dir, frames_root, fps=args.fps, debug=args.debug)
        else:
            print(f"[ERROR] No mp4 videos found in: {videos_dir}")
            print("[INFO] Provide --videos_dir pointing to a directory with mp4 files.")
            return

    if args.step in ("all", "metadata"):
        print("\n[STEP 2] Building frame-level metadata...")
        metadata_by_split = build_frame_level_metadata(
            frames_root=frames_root,
            label_csv=label_csv,
            output_dir=output_dir,
            debug=args.debug,
        )
        save_metadata_csvs(output_dir, metadata_by_split)

    if args.step in ("all", "clips"):
        print(f"\n[STEP 3] Generating dense clips (window_size={args.window_size})...")
        generate_dense_clips(
            base_data_path=str(output_dir),
            window_size=args.window_size,
            stride=args.stride,
            fps=args.fps,
            enable_padding=not args.no_padding,
        )

    print("\n" + "=" * 60)
    print("OphNet2024 preprocessing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
