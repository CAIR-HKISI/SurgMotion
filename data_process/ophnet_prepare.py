#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OphNet2024 phase recognition preprocessing (end-to-end pipeline style).

Expected layout
---------------
- Phase interval CSV (challenge split): columns ``video_id``, ``start``, ``end``,
  ``phase_id``, ``split`` (train / val / test).
- Extracted frames per interval (clip):
  ``<frames_root>/{video_id}_{clip_index}/*.jpg``
  where ``clip_index`` is a per-``video_id`` cumulative index after sorting rows by ``start``.

Pipeline
--------
1. ``videos_to_frames()`` — stub; frames are normally pre-extracted to match the CSV clips.
2. ``generate_clip_txt()`` — for each clip folder, write ``<output_dir>/clip_infos/{video_id}_{clip_index}.txt``
   listing frame paths (sorted by filename).
3. ``build_metadata()`` — read the label CSV, assign ``clip_index``, emit **one metadata row per frame**
   with ``label`` = ``phase_id``, ``label_name`` from the phase map, ``case_id`` from ``video_id`` (``case_`` prefix
   stripped), ``clip_idx`` = ``clip_index``.

Outputs (under ``--output_dir``)
--------------------------------
- ``clip_infos/{video_id}_{clip_index}.txt``
- ``train_metadata.csv``, ``val_metadata.csv``, ``test_metadata.csv``
- ``metadata.csv`` (all splits combined)
- ``missing_frames_report.csv`` (missing clip directories or empty folders)

CSV columns: ``Index``, ``clip_path``, ``label``, ``label_name``, ``case_id``, ``clip_idx``
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

# 96 phases (0–95)
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
    Stub: extract frames from raw OphNet videos into clip folders under ``output_path``.

    Implement when raw videos and official clip boundaries are wired to ffmpeg; for the
    released frame dump, point ``--frames_root`` at the existing ``.../frames`` tree instead.
    """
    print(
        "videos_to_frames: stub — no extraction run. "
        "Use pre-extracted frames under --frames_root or implement extraction here."
    )
    if debug:
        print(f"  input_path={input_path}, output_path={output_path}, fps={fps}")


def _list_jpeg_frames(video_frames_dir: Path) -> List[Path]:
    """Sorted ``.jpg`` / ``.jpeg`` files under a clip directory."""
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
    Write one line per JPEG frame under ``video_frames_dir`` (sorted by filename).
    Paths use forward slashes (relative to the project root / cwd, same as other prepare scripts).
    Returns the number of lines written.
    """
    if not video_frames_dir.is_dir():
        return 0

    frame_files = _list_jpeg_frames(video_frames_dir)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with txt_path.open("w", encoding="utf-8") as f:
        for frame_path in frame_files:
            f.write(str(frame_path).replace("\\", "/") + "\n")
    return len(frame_files)


def _parse_case_id(video_id: str) -> int:
    """Strip optional ``case_`` prefix and parse integer case id."""
    vid = str(video_id).strip()
    if vid.startswith("case_"):
        vid = vid[5:]
    return int(vid)


def build_metadata(
    frames_root: Path,
    label_csv: Path,
    output_dir: Path,
    debug: bool = False,
) -> Tuple[List[dict], List[dict], List[dict], List[dict], List[dict]]:
    """
    Read the label CSV, compute ``clip_index`` per ``video_id``, write clip txts, and build
    per-frame metadata rows (same ``clip_path`` txt for all frames in one clip).

    Returns ``(train_rows, val_rows, test_rows, all_rows_ordered, missing_report)``.
    """
    if not label_csv.is_file():
        raise FileNotFoundError(f"Label CSV not found: {label_csv}")

    df = pd.read_csv(label_csv)
    required = {"video_id", "start", "phase_id", "split"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Label CSV missing columns {missing_cols}; got {list(df.columns)}")

    df = df.sort_values(["video_id", "start"]).reset_index(drop=True)
    df["clip_index"] = df.groupby("video_id", sort=False).cumcount()

    clip_infos_dir = output_dir / "clip_infos"
    clip_infos_dir.mkdir(parents=True, exist_ok=True)

    train_rows: List[dict] = []
    val_rows: List[dict] = []
    test_rows: List[dict] = []
    all_rows: List[dict] = []
    missing_report: List[dict] = []

    global_index = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="OphNet2024 clips"):
        video_id = str(row["video_id"]).strip()
        clip_index = int(row["clip_index"])
        split_raw = str(row["split"]).strip().lower()
        phase_id = int(row["phase_id"])
        label_name = PHASE2NAME.get(phase_id, "Unknown Phase")

        if split_raw not in ("train", "val", "test"):
            missing_report.append(
                {
                    "video_id": video_id,
                    "clip_idx": clip_index,
                    "split": split_raw,
                    "reason": "unknown_split",
                    "detail": "",
                }
            )
            if debug:
                print(f"Skipping row: unknown split {split_raw!r} for {video_id} clip {clip_index}")
            continue

        try:
            case_id = _parse_case_id(video_id)
        except ValueError:
            missing_report.append(
                {
                    "video_id": video_id,
                    "clip_idx": clip_index,
                    "split": split_raw,
                    "reason": "bad_video_id",
                    "detail": "cannot parse integer case_id",
                }
            )
            if debug:
                print(f"Skipping row: cannot parse case_id from video_id={video_id!r}")
            continue

        clip_key = f"{video_id}_{clip_index}"
        clip_dir = frames_root / clip_key
        txt_path = clip_infos_dir / f"{clip_key}.txt"

        if not clip_dir.is_dir():
            missing_report.append(
                {
                    "video_id": video_id,
                    "clip_idx": clip_index,
                    "split": split_raw,
                    "reason": "missing_clip_directory",
                    "detail": str(clip_dir).replace("\\", "/"),
                }
            )
            if debug:
                print(f"Missing clip directory: {clip_dir}")
            continue

        n_frames = generate_clip_txt(clip_dir, txt_path)
        if n_frames == 0:
            missing_report.append(
                {
                    "video_id": video_id,
                    "clip_idx": clip_index,
                    "split": split_raw,
                    "reason": "empty_clip_directory",
                    "detail": str(clip_dir).replace("\\", "/"),
                }
            )
            if debug:
                print(f"No frames in {clip_dir}")
            continue

        clip_path_str = str(txt_path).replace("\\", "/")

        frame_files = _list_jpeg_frames(clip_dir)

        for _ in frame_files:
            item = {
                "Index": global_index,
                "clip_path": clip_path_str,
                "label": phase_id,
                "label_name": label_name,
                "case_id": case_id,
                "clip_idx": clip_index,
            }
            global_index += 1
            all_rows.append(item)

            if split_raw == "train":
                train_rows.append(item.copy())
            elif split_raw == "val":
                val_rows.append(item.copy())
            else:
                test_rows.append(item.copy())

    return train_rows, val_rows, test_rows, all_rows, missing_report


def _save_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"Saved {len(rows)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OphNet2024: clip txts + per-frame phase metadata (train/val/test)."
    )
    parser.add_argument(
        "--frames_root",
        type=str,
        default="data/Surge_Frames/OphNet2024_phase/frames",
        help="Root containing clip folders {video_id}_{clip_index}/ with .jpg frames.",
    )
    parser.add_argument(
        "--label_csv",
        type=str,
        default="data/Ophthalmology/OphNet2024_trimmed_phase/OphNet2024_loca_challenge_phase.csv",
        help="CSV with video_id, start, end, phase_id, split.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/Surge_Frames/OphNet2024_phase",
        help="Output directory for clip_infos/, metadata CSVs, missing_frames_report.csv.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print diagnostics for skipped rows and stub extraction.",
    )
    args = parser.parse_args()

    frames_root = Path(args.frames_root)
    label_csv = Path(args.label_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Uncomment and set paths when extracting from raw videos:
    # videos_to_frames(Path("path/to/raw/videos"), frames_root, debug=args.debug)

    train_rows, val_rows, test_rows, all_rows, missing_report = build_metadata(
        frames_root=frames_root,
        label_csv=label_csv,
        output_dir=output_dir,
        debug=args.debug,
    )

    _save_csv(output_dir / "train_metadata.csv", train_rows)
    _save_csv(output_dir / "val_metadata.csv", val_rows)
    _save_csv(output_dir / "test_metadata.csv", test_rows)
    _save_csv(output_dir / "metadata.csv", all_rows)

    miss_path = output_dir / "missing_frames_report.csv"
    miss_path.parent.mkdir(parents=True, exist_ok=True)
    if missing_report:
        pd.DataFrame(missing_report).to_csv(miss_path, index=False)
        print(f"Wrote missing / skipped clip report ({len(missing_report)} rows): {miss_path}")
    else:
        pd.DataFrame(
            columns=["video_id", "clip_idx", "split", "reason", "detail"],
        ).to_csv(miss_path, index=False)
        print(f"No issues logged; wrote empty report: {miss_path}")

    print("OphNet2024 preprocessing done.")


if __name__ == "__main__":
    main()
