#!/usr/bin/env python3
"""
Bleeding 数据集预处理 Pipeline V4
每个 3s/30fps 视频降采样至 20fps（60 帧）独立作为一个 case，
按病人级别划分 train/val/test，并输出 patient-level metadata 用于追踪。

用法:
    python data_process/bleeding_prepare_V4.py                                # 完整 pipeline
    python data_process/bleeding_prepare_V4.py --out_dir data/Surge_Frames/Bleeding_V5_5fps_70_15_15
    python data_process/bleeding_prepare_V4.py --split_ratios 0.70,0.15,0.15
    python data_process/bleeding_prepare_V4.py --split_ratios 80,20,0         # 百分比形式
    python data_process/bleeding_prepare_V4.py --step parse                   # 仅解析文件名
    python data_process/bleeding_prepare_V4.py --step extract                 # 仅抽帧
    python data_process/bleeding_prepare_V4.py --step split                   # 仅划分
    python data_process/bleeding_prepare_V4.py --step csv                     # 仅生成 CSV
"""

import os
import re
import json
import argparse
import subprocess
from collections import defaultdict
from urllib.parse import unquote

import numpy as np
import pandas as pd
from tqdm import tqdm

# ========================= #
# 配置（可由命令行覆盖）
# ========================= #
RAW_ROOT = "data/raw_bleeding"
OUT_DIR = "data/Surge_Frames/Bleeding_V5_5fps"
DEFAULT_SPLIT_RATIOS = (0.70, 0.30, 0.00)

DATA_NAME = "Bleeding_Dataset"
YEAR = 2025
FPS_TARGET = 5
SEED = 42

CATEGORIES = {
    "Bleeding":        {"label": 1, "label_name": "bleeding"},
    "Non_bleeding":    {"label": 0, "label_name": "non_bleeding"},
    "new_Nonbleeding": {"label": 0, "label_name": "non_bleeding"},
}


# ========================= #
# Step 1: 文件名解析 + 独立 Case 分配
# ========================= #

def parse_bleeding_filename(filename: str) -> dict | None:
    """case_110_Bleeding_0_02_57.344011-0_03_07.78778830fps_clip_1.mp4"""
    m = re.match(
        r"case_(\d+)_Bleeding_(?:light_)?"
        r"(\d+)_(\d+)_([\d.]+)-(\d+)_(\d+)_([\d.]+)30fps_clip_?(\d+)\.mp4",
        filename,
    )
    if not m:
        return None
    cid, h1, m1, s1, h2, m2, s2, clip_id = m.groups()
    start = int(h1) * 3600 + int(m1) * 60 + float(s1)
    return {
        "case_id": int(cid),
        "patient_id": f"Bleeding_{cid}",
        "start_sec": start,
        "clip_id": int(clip_id),
    }


def parse_non_bleeding_filename(filename: str) -> dict | None:
    """case_103_Non_bleeding_0_08_08.288288-0_08_14.01067730fps_clip_2.mp4"""
    m = re.match(
        r"case_(\d+)_Non_bleeding_(?:bleeding_)?"
        r"(\d+)_(\d+)_([\d.]+)-(\d+)_(\d+)_([\d.]+)30fps_clip_?(\d+)\.mp4",
        filename,
    )
    if not m:
        return None
    cid, h1, m1, s1, h2, m2, s2, clip_id = m.groups()
    start = int(h1) * 3600 + int(m1) * 60 + float(s1)
    return {
        "case_id": int(cid),
        "patient_id": f"NonBleeding_{cid}",
        "start_sec": start,
        "clip_id": int(clip_id),
    }


def parse_new_nonbleeding_filename(filename: str) -> dict | None:
    """case_1101_0%3A00%3A00-0%3A01%3A00_clip_0.mp4
    yyzz: yy = patient_id, zz = video within patient
    """
    m_case = re.match(r"case_(\d{2})(\d{2})_", filename)
    if not m_case:
        return None
    yy, zz = m_case.group(1), m_case.group(2)

    m_time = re.match(
        r"case_\d+_(\d+)%3A(\d+)%3A(\d+)-(\d+)%3A(\d+)%3A(\d+)_clip_(\d+)\.mp4",
        filename,
    )
    if not m_time:
        return None
    h1, m1, s1, h2, m2, s2, clip_id = m_time.groups()
    start = int(h1) * 3600 + int(m1) * 60 + int(s1)
    return {
        "case_id": int(yy + zz),
        "patient_id": f"NNB_{yy}",
        "start_sec": start,
        "clip_id": int(clip_id),
    }


PARSERS = {
    "Bleeding": parse_bleeding_filename,
    "Non_bleeding": parse_non_bleeding_filename,
    "new_Nonbleeding": parse_new_nonbleeding_filename,
}


def collect_all_clips(raw_root: str) -> list[dict]:
    """Scan all three category dirs, parse every mp4 filename, return list of clip dicts."""
    clips = []
    for cat, meta in CATEGORIES.items():
        cat_dir = os.path.join(raw_root, cat)
        if not os.path.isdir(cat_dir):
            print(f"WARNING: {cat_dir} not found, skipping")
            continue
        parser = PARSERS[cat]
        mp4s = sorted(f for f in os.listdir(cat_dir) if f.endswith(".mp4"))
        unparsed = 0
        for fname in mp4s:
            info = parser(fname)
            if info is None:
                unparsed += 1
                continue
            info["category"] = cat
            info["label"] = meta["label"]
            info["label_name"] = meta["label_name"]
            info["filepath"] = os.path.join(cat_dir, fname)
            info["filename"] = fname
            clips.append(info)
        if unparsed:
            print(f"WARNING: {unparsed} files in {cat} could not be parsed")
    print(f"Parsed {len(clips)} clips total")
    return clips


def assign_video_cases(clips: list[dict]) -> dict:
    """
    Treat each clip as an independent case.
    Group by patient_id, sort within each patient by (start_sec, clip_id),
    and assign sequential case_name: {patient_id}_case_{N} (N starts at 1).

    Returns {case_name: {"clip": clip_dict, "label": int, "label_name": str, "patient_id": str}}
    """
    by_patient = defaultdict(list)
    for c in clips:
        by_patient[c["patient_id"]].append(c)

    cases = {}
    for pid in sorted(by_patient.keys()):
        patient_clips = by_patient[pid]
        patient_clips.sort(key=lambda x: (x["start_sec"], x["clip_id"]))
        for idx, clip in enumerate(patient_clips, start=1):
            case_name = f"{pid}_case_{idx}"
            cases[case_name] = {
                "clip": clip,
                "label": clip["label"],
                "label_name": clip["label_name"],
                "patient_id": pid,
            }

    print(f"Assigned {len(cases)} independent video cases "
          f"from {len(by_patient)} patients")
    return cases


# ========================= #
# Step 2: 抽帧 (30fps -> 20fps, 60 frames/video)
# ========================= #

def extract_frames_for_video(case_name: str, case_info: dict, frame_dir: str) -> int:
    """
    Extract 20fps frames from a single 3s clip.
    Returns the number of frames extracted.
    """
    out_folder = os.path.join(frame_dir, case_name)
    os.makedirs(out_folder, exist_ok=True)

    clip = case_info["clip"]
    tmp_dir = os.path.join(out_folder, "_tmp_clip")
    os.makedirs(tmp_dir, exist_ok=True)

    ffmpeg_cmd = (
        f'ffmpeg -y -i "{clip["filepath"]}" '
        f"-vf \"fps={FPS_TARGET},"
        f"scale='if(gte(iw,ih),512,-1)':'if(gte(ih,iw),512,-1)':"
        f'force_original_aspect_ratio=decrease" '
        f'"{tmp_dir}/frame_%08d.jpg" '
        f"-loglevel error"
    )
    subprocess.run(ffmpeg_cmd, shell=True, check=False)

    tmp_frames = sorted(f for f in os.listdir(tmp_dir) if f.endswith(".jpg"))
    for i, tf in enumerate(tmp_frames, start=1):
        src = os.path.join(tmp_dir, tf)
        dst = os.path.join(out_folder, f"{case_name}_{i:08d}.jpg")
        os.rename(src, dst)

    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    return len(tmp_frames)


def run_extraction(cases: dict, frame_dir: str):
    """Extract frames for all video cases."""
    os.makedirs(frame_dir, exist_ok=True)
    total_frames = 0
    for case_name, case_info in tqdm(cases.items(), desc="Extracting frames"):
        existing = os.path.join(frame_dir, case_name)
        if os.path.isdir(existing) and len(os.listdir(existing)) > 0:
            n = len([f for f in os.listdir(existing) if f.endswith(".jpg")])
            total_frames += n
            continue
        n = extract_frames_for_video(case_name, case_info, frame_dir)
        total_frames += n
    print(f"Total frames extracted: {total_frames}")


# ========================= #
# Step 3: Patient-based train/val/test split
# ========================= #

def split_by_patient(
    cases: dict,
    split_ratios: tuple = DEFAULT_SPLIT_RATIOS,
    seed: int = SEED,
) -> tuple[dict, dict]:
    """
    Split cases into train/val/test by patient_id.
    Returns (case_split, patient_metadata):
      - case_split: {case_name: split_name}
      - patient_metadata: {patient_id: {"split": str, "videos": [...]}}
    """
    rng = np.random.RandomState(seed)

    patient_to_cases = defaultdict(list)
    for case_name, info in cases.items():
        patient_to_cases[info["patient_id"]].append(case_name)

    patients = sorted(patient_to_cases.keys())
    rng.shuffle(patients)

    n = len(patients)
    n_train = int(n * split_ratios[0])
    n_val = int(n * split_ratios[1])

    split_labels = ["train"] * n_train + ["val"] * n_val
    split_labels += ["test"] * (n - len(split_labels))

    case_split = {}
    patient_metadata = {}
    for pid, sp in zip(patients, split_labels):
        case_names = sorted(patient_to_cases[pid])
        for cn in case_names:
            case_split[cn] = sp
        patient_metadata[pid] = {
            "patient_id": pid,
            "split": sp,
            "videos": [
                {
                    "case_name": cn,
                    "source_file": os.path.basename(cases[cn]["clip"]["filepath"]),
                }
                for cn in case_names
            ],
        }

    split_counts = defaultdict(lambda: {"cases": 0, "patients": 0, "pos": 0, "neg": 0})
    for pid, meta in patient_metadata.items():
        sp = meta["split"]
        split_counts[sp]["patients"] += 1
        for v in meta["videos"]:
            split_counts[sp]["cases"] += 1
            if cases[v["case_name"]]["label"] == 1:
                split_counts[sp]["pos"] += 1
            else:
                split_counts[sp]["neg"] += 1

    print(f"\nPatient-based split (seed={seed}):")
    print(f"  Total patients: {n}")
    for sp in ["train", "val", "test"]:
        c = split_counts[sp]
        print(f"  {sp}: {c['patients']} patients, {c['cases']} cases "
              f"(pos={c['pos']}, neg={c['neg']})")

    return case_split, patient_metadata


# ========================= #
# Step 4: 生成帧级 CSV
# ========================= #

def generate_frame_csvs(cases: dict, case_split: dict, frame_dir: str, out_dir: str):
    """Generate {split}_metadata.csv files."""
    os.makedirs(out_dir, exist_ok=True)

    split_data = defaultdict(list)
    global_idx = 0
    case_id_map = {}
    next_case_id = 1

    for case_name in sorted(cases.keys()):
        if case_name not in case_split:
            continue

        split = case_split[case_name]
        info = cases[case_name]

        if case_name not in case_id_map:
            case_id_map[case_name] = next_case_id
            next_case_id += 1
        case_id = case_id_map[case_name]

        case_frame_dir = os.path.join(frame_dir, case_name)
        if not os.path.isdir(case_frame_dir):
            print(f"WARNING: frame dir not found: {case_frame_dir}")
            continue

        frame_files = sorted(
            f for f in os.listdir(case_frame_dir) if f.endswith(".jpg")
        )
        if not frame_files:
            print(f"WARNING: no frames in {case_frame_dir}")
            continue

        for ff in frame_files:
            frame_path = os.path.join(frame_dir, case_name, ff)
            split_data[split].append({
                "index": global_idx,
                "DataName": DATA_NAME,
                "Year": YEAR,
                "Case_Name": case_name,
                "Case_ID": case_id,
                "Frame_Path": frame_path,
                "Phase_GT": info["label"],
                "Phase_Name": info["label_name"],
                "Split": split,
            })
            global_idx += 1

    for split_name in ["train", "val", "test"]:
        items = split_data.get(split_name, [])
        if not items:
            print(f"WARNING: no data for split {split_name}")
            continue
        df = pd.DataFrame(items)
        csv_path = os.path.join(out_dir, f"{split_name}_metadata.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(df)} frames -> {csv_path}")


# ========================= #
# 主程序
# ========================= #

def parse_split_ratios(s: str) -> tuple:
    """Parse '0.70,0.15,0.15' or '70,15,15' -> (0.70, 0.15, 0.15)"""
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError("--split_ratios 需要 3 个数字，如 0.70,0.15,0.15 或 70,15,15")
    if 99 <= sum(parts) <= 101:
        return tuple(x / 100.0 for x in parts)
    return tuple(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Bleeding dataset preparation pipeline V4 "
                    "(per-video case, 20fps)"
    )
    parser.add_argument("--raw_root", type=str, default=RAW_ROOT,
                        help="原始 mp4 根目录")
    parser.add_argument("--out_dir", type=str, default=OUT_DIR,
                        help="输出目录，帧与 CSV 将写入此处")
    parser.add_argument("--split_ratios", type=str, default="0.70,0.30,0.00",
                        help="train,val,test 比例，如 0.70,0.15,0.15 或 70,15,15")
    parser.add_argument("--step", type=str, default="all",
                        choices=["all", "parse", "extract", "split", "csv"],
                        help="Run a specific step or all")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    split_ratios = parse_split_ratios(args.split_ratios)
    frame_dir = os.path.join(args.out_dir, "frames")
    cache_file = os.path.join(args.out_dir, "_case_cache.json")

    # ---- Step 1: Parse & assign per-video cases ----
    if args.step in ("all", "parse"):
        print("=" * 60)
        print("Step 1: Parsing filenames & assigning per-video cases")
        print("=" * 60)
        clips = collect_all_clips(args.raw_root)
        cases = assign_video_cases(clips)

        os.makedirs(args.out_dir, exist_ok=True)
        serializable = {}
        for cn, info in cases.items():
            clip = info["clip"]
            serializable[cn] = {
                "label": info["label"],
                "label_name": info["label_name"],
                "patient_id": info["patient_id"],
                "clip": {
                    "filepath": clip["filepath"],
                    "filename": clip["filename"],
                    "start_sec": clip["start_sec"],
                    "clip_id": clip["clip_id"],
                },
            }
        with open(cache_file, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"Case info cached -> {cache_file}")

    # Load cache for subsequent steps
    if args.step != "parse":
        if not os.path.exists(cache_file):
            print(f"ERROR: cache file not found: {cache_file}. "
                  f"Run with --step parse first.")
            return
        with open(cache_file) as f:
            cases = json.load(f)

    # ---- Step 2: Extract frames (30fps -> 20fps) ----
    if args.step in ("all", "extract"):
        print("\n" + "=" * 60)
        print("Step 2: Extracting frames (30fps -> 20fps, ~60 frames/video)")
        print("=" * 60)
        run_extraction(cases, frame_dir)

    # ---- Step 3: Patient-based split ----
    if args.step in ("all", "split", "csv"):
        print("\n" + "=" * 60)
        print("Step 3: Patient-based train/val/test split")
        print("=" * 60)
        case_split, patient_metadata = split_by_patient(
            cases, split_ratios=split_ratios, seed=args.seed
        )

        split_file = os.path.join(args.out_dir, "case_split.json")
        with open(split_file, "w") as f:
            json.dump(case_split, f, indent=2)
        print(f"Split mapping saved -> {split_file}")

        meta_file = os.path.join(args.out_dir, "patient_video_metadata.json")
        with open(meta_file, "w") as f:
            json.dump(patient_metadata, f, indent=2, ensure_ascii=False)
        print(f"Patient metadata saved -> {meta_file}")

    # ---- Step 4: Generate frame-level CSVs ----
    if args.step in ("all", "csv"):
        print("\n" + "=" * 60)
        print("Step 4: Generating frame-level CSVs")
        print("=" * 60)
        split_file = os.path.join(args.out_dir, "case_split.json")
        if not os.path.exists(split_file):
            print(f"ERROR: split file not found: {split_file}. "
                  f"Run with --step split first.")
            return
        with open(split_file) as f:
            case_split = json.load(f)
        generate_frame_csvs(cases, case_split, frame_dir, args.out_dir)

    print("\nPipeline V4 complete!")


if __name__ == "__main__":
    main()
