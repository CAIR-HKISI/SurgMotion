#!/usr/bin/env python3
"""
Bleeding 数据集预处理 Pipeline
将 raw mp4 片段（30fps）转换为 SurgFrames 标准格式（1fps 帧 + 帧级 CSV）

用法:
    python data_process/bleeding_prepare.py                          # 完整 pipeline
    python data_process/bleeding_prepare.py --out_dir data/Surge_Frames/Bleeding_70_30 --split_ratios 0.70,0.30,0
    python data_process/bleeding_prepare.py --split_ratios 80,20,0   # 百分比形式
    python data_process/bleeding_prepare.py --step parse             # 仅解析文件名
    python data_process/bleeding_prepare.py --step extract           # 仅抽帧
    python data_process/bleeding_prepare.py --step split             # 仅划分 train/val/test
    python data_process/bleeding_prepare.py --step csv               # 仅生成 CSV
"""

import os
import re
import json
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict
from urllib.parse import unquote

import numpy as np
import pandas as pd
from tqdm import tqdm

# ========================= #
# 配置（可由命令行覆盖）
# ========================= #
RAW_ROOT = "data/raw_bleeding"
OUT_DIR = "data/Surge_Frames/Bleeding_Dataset_80_20"  # 默认值，可用 --out_dir 覆盖
DEFAULT_SPLIT_RATIOS = (0.80, 0.20, 0)  # train / val / test，可用 --split_ratios 覆盖

DATA_NAME = "Bleeding_Dataset"
YEAR = 2025
FPS_TARGET = 1
SEED = 42

CATEGORIES = {
    "Bleeding":        {"label": 1, "label_name": "bleeding"},
    "Non_bleeding":    {"label": 0, "label_name": "non_bleeding"},
    "new_Nonbleeding": {"label": 0, "label_name": "non_bleeding"},
}


# ========================= #
# Step 1: 文件名解析与 Case 聚合
# ========================= #

def parse_bleeding_filename(filename: str) -> dict | None:
    """case_110_Bleeding_0_02_57.344011-0_03_07.78778830fps_clip_1.mp4
    or:  case_192_Bleeding_light_0_01_16.647948-0_01_56.49284730fps_clip_37.mp4
    or:  case_659_Bleeding_0_05_11.560000-0_06_37.12000030fps_clip1.mp4  (no underscore before clip num)
    """
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
        "case_name": f"Bleeding_{cid}",
        "start_sec": start,
        "clip_id": int(clip_id),
    }


def parse_non_bleeding_filename(filename: str) -> dict | None:
    """case_103_Non_bleeding_0_08_08.288288-0_08_14.01067730fps_clip_2.mp4
    or:  case_668_Non_bleeding_bleeding_0_22_01.160000-0_22_04.80000030fps_clip_1.mp4
    """
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
        "case_name": f"NonBleeding_{cid}",
        "start_sec": start,
        "clip_id": int(clip_id),
    }


def parse_new_nonbleeding_filename(filename: str) -> dict | None:
    """case_1101_0%3A00%3A00-0%3A01%3A00_clip_0.mp4
    yyzz: yy = patient_id, zz = video within patient
    """
    decoded = unquote(filename)  # %3A -> :
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
        "case_name": f"NNB_{yy}{zz}",
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
            clips.append(info)
        if unparsed:
            print(f"WARNING: {unparsed} files in {cat} could not be parsed")
    print(f"Parsed {len(clips)} clips total")
    return clips


def aggregate_cases(clips: list[dict]) -> dict:
    """
    Group clips by case_name, sort within each case by (start_sec, clip_id).
    Returns {case_name: {"clips": [...], "label": int, "label_name": str, "patient_id": str}}
    """
    groups = defaultdict(list)
    for c in clips:
        groups[c["case_name"]].append(c)

    cases = {}
    for case_name, case_clips in sorted(groups.items()):
        case_clips.sort(key=lambda x: (x["start_sec"], x["clip_id"]))
        cases[case_name] = {
            "clips": case_clips,
            "label": case_clips[0]["label"],
            "label_name": case_clips[0]["label_name"],
            "patient_id": case_clips[0]["patient_id"],
        }
    print(f"Aggregated into {len(cases)} unique cases")
    return cases


# ========================= #
# Step 2: 抽帧 (30fps -> 1fps)
# ========================= #

def extract_frames_for_case(case_name: str, case_info: dict, frame_dir: str) -> int:
    """
    Extract 1fps frames from all clips of a case, with global sequential numbering.
    Returns the total number of frames extracted for this case.
    """
    out_folder = os.path.join(frame_dir, case_name)
    os.makedirs(out_folder, exist_ok=True)

    global_frame_idx = 1
    for clip in case_info["clips"]:
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

        tmp_frames = sorted(
            f for f in os.listdir(tmp_dir) if f.endswith(".jpg")
        )
        for tf in tmp_frames:
            src = os.path.join(tmp_dir, tf)
            dst = os.path.join(out_folder, f"{case_name}_{global_frame_idx:08d}.jpg")
            os.rename(src, dst)
            global_frame_idx += 1

        # clean up tmp
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass

    return global_frame_idx - 1


def run_extraction(cases: dict, frame_dir: str):
    """Extract frames for all cases."""
    os.makedirs(frame_dir, exist_ok=True)
    total_frames = 0
    for case_name, case_info in tqdm(cases.items(), desc="Extracting frames"):
        existing = os.path.join(frame_dir, case_name)
        if os.path.isdir(existing) and len(os.listdir(existing)) > 0:
            n = len([f for f in os.listdir(existing) if f.endswith(".jpg")])
            total_frames += n
            continue
        n = extract_frames_for_case(case_name, case_info, frame_dir)
        total_frames += n
    print(f"Total frames extracted: {total_frames}")


# ========================= #
# Step 3: Patient-based train/val/test split (70/15/15)
# ========================= #

def split_by_patient(cases: dict, split_ratios: tuple = DEFAULT_SPLIT_RATIOS, seed: int = SEED) -> dict:
    """
    Split cases into train/val/test by patient_id.
    split_ratios: (train, val, test) e.g. (0.80, 0.20, 0) or (0.70, 0.15, 0.15).
    Same patient's cases never span splits.
    Returns {case_name: split_name}
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

    train_patients = set(patients[:n_train])
    val_patients = set(patients[n_train:n_train + n_val])
    test_patients = set(patients[n_train + n_val:])

    case_split = {}
    for pid in train_patients:
        for cn in patient_to_cases[pid]:
            case_split[cn] = "train"
    for pid in val_patients:
        for cn in patient_to_cases[pid]:
            case_split[cn] = "val"
    for pid in test_patients:
        for cn in patient_to_cases[pid]:
            case_split[cn] = "test"

    # summary
    split_counts = defaultdict(lambda: {"cases": 0, "pos": 0, "neg": 0})
    for cn, sp in case_split.items():
        split_counts[sp]["cases"] += 1
        if cases[cn]["label"] == 1:
            split_counts[sp]["pos"] += 1
        else:
            split_counts[sp]["neg"] += 1

    print(f"\nPatient-based split (seed={seed}):")
    print(f"  Total patients: {n} | train: {len(train_patients)} | val: {len(val_patients)} | test: {len(test_patients)}")
    for sp in ["train", "val", "test"]:
        c = split_counts[sp]
        print(f"  {sp}: {c['cases']} cases (pos={c['pos']}, neg={c['neg']})")

    return case_split


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
    """Parse '0.80,0.20,0' or '80,20,0' -> (0.80, 0.20, 0.0)"""
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError("--split_ratios 需要 3 个数字，如 0.80,0.20,0 或 80,20,0")
    if sum(parts) >= 99 and sum(parts) <= 101:
        # 百分比形式
        return tuple(x / 100.0 for x in parts)
    return tuple(parts)


def main():
    parser = argparse.ArgumentParser(description="Bleeding dataset preparation pipeline")
    parser.add_argument("--raw_root", type=str, default=RAW_ROOT, help="原始 mp4 根目录")
    parser.add_argument("--out_dir", type=str, default=OUT_DIR,
                        help="输出目录，帧与 CSV 将写入此处")
    parser.add_argument("--split_ratios", type=str, default="0.80,0.20,0",
                        help="train,val,test 比例，如 0.80,0.20,0 或 80,20,0")
    parser.add_argument("--step", type=str, default="all",
                        choices=["all", "parse", "extract", "split", "csv"],
                        help="Run a specific step or all")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    split_ratios = parse_split_ratios(args.split_ratios)
    frame_dir = os.path.join(args.out_dir, "frames")
    cache_file = os.path.join(args.out_dir, "_case_cache.json")

    # Step 1: Parse & aggregate
    if args.step in ("all", "parse"):
        print("=" * 60)
        print("Step 1: Parsing filenames & aggregating cases")
        print("=" * 60)
        clips = collect_all_clips(args.raw_root)
        cases = aggregate_cases(clips)

        os.makedirs(args.out_dir, exist_ok=True)
        serializable = {}
        for cn, info in cases.items():
            serializable[cn] = {
                "label": info["label"],
                "label_name": info["label_name"],
                "patient_id": info["patient_id"],
                "clips": [
                    {"filepath": c["filepath"], "start_sec": c["start_sec"], "clip_id": c["clip_id"]}
                    for c in info["clips"]
                ],
            }
        with open(cache_file, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"Case info cached -> {cache_file}")

    # Load cache for subsequent steps
    if args.step != "parse":
        if not os.path.exists(cache_file):
            print(f"ERROR: cache file not found: {cache_file}. Run with --step parse first.")
            return
        with open(cache_file) as f:
            cases = json.load(f)

    # Step 2: Extract frames
    if args.step in ("all", "extract"):
        print("\n" + "=" * 60)
        print("Step 2: Extracting frames (30fps -> 1fps)")
        print("=" * 60)
        run_extraction(cases, frame_dir)

    # Step 3: Split
    if args.step in ("all", "split", "csv"):
        print("\n" + "=" * 60)
        print("Step 3: Patient-based train/val/test split")
        print("=" * 60)
        case_split = split_by_patient(cases, split_ratios=split_ratios, seed=args.seed)

        split_file = os.path.join(args.out_dir, "case_split.json")
        with open(split_file, "w") as f:
            json.dump(case_split, f, indent=2)
        print(f"Split mapping saved -> {split_file}")

    # Step 4: Generate frame-level CSVs
    if args.step in ("all", "csv"):
        print("\n" + "=" * 60)
        print("Step 4: Generating frame-level CSVs")
        print("=" * 60)
        split_file = os.path.join(args.out_dir, "case_split.json")
        if not os.path.exists(split_file):
            print(f"ERROR: split file not found: {split_file}. Run with --step split first.")
            return
        with open(split_file) as f:
            case_split = json.load(f)
        generate_frame_csvs(cases, case_split, frame_dir, args.out_dir)

    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
