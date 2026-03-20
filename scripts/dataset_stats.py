#!/usr/bin/env python3
"""
Bleeding Dataset 统计脚本
用法: python scripts/dataset_stats.py [数据集目录] [选项]

示例:
  python scripts/dataset_stats.py data/Surge_Frames/Bleeding_Dataset_70_30
  python scripts/dataset_stats.py data/Surge_Frames/Bleeding_Dataset
  python scripts/dataset_stats.py data/Surge_Frames/Bleeding_Dataset_70_30 --list
  python scripts/dataset_stats.py data/Surge_Frames/Bleeding_Dataset_70_30 --json
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path


def categorize_case(case_name: str) -> str:
    """根据 case 名称判断原始类别"""
    if case_name.startswith("Bleeding_"):
        return "Bleeding"
    elif case_name.startswith("NonBleeding_"):
        return "NonBleeding"
    elif case_name.startswith("NNB_"):
        return "new_Nonbleeding"
    return "Unknown"


def count_frames_by_label(csv_path: str) -> tuple[Counter, set]:
    """统计 metadata CSV 中的帧数和标签分布"""
    labels = Counter()
    cases = set()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[int(row["Phase_GT"])] += 1
            cases.add(row["Case_Name"])
    return labels, cases


def print_section(title: str, width: int = 70):
    """打印分节标题"""
    print(f"\n{'=' * width}")
    print(f" {title}")
    print(f"{'=' * width}")


def print_patient_list(splits: list, patient_list_by_split_cat: dict, case_split: dict):
    """打印病人编号划分详情"""
    print_section("病人编号划分详情")

    # 按 split 分组显示
    for split in splits:
        print(f"\n[{split.upper()}] 共 {sum(len(v) for v in patient_list_by_split_cat[split].values())} 个病人")
        print("-" * 70)

        cats = sorted(patient_list_by_split_cat[split].keys())
        for cat in cats:
            patients = patient_list_by_split_cat[split][cat]
            print(f"\n  {cat} ({len(patients)} 个):")
            # 每行显示多个编号
            line = "    "
            for i, pid in enumerate(patients):
                if len(line) + len(pid) + 2 > 70:
                    print(line)
                    line = "    "
                line += f"{pid}, "
            if line.strip():
                print(line.rstrip(", "))

    # 汇总表格
    print_section("按类别汇总")
    all_cats = sorted(set(c for s in patient_list_by_split_cat.values() for c in s.keys()))

    # 表头
    header = f"{'类别':<20}"
    for split in splits:
        header += f" {split:>10}"
    header += f" {'合计':>10}"
    print(header)
    print("-" * (22 + 12 * (len(splits) + 1)))

    # 每个类别一行
    for cat in all_cats:
        row = f"{cat:<20}"
        row_total = 0
        for split in splits:
            cnt = len(patient_list_by_split_cat[split].get(cat, []))
            row_total += cnt
            row += f" {cnt:>10}"
        row += f" {row_total:>10}"
        print(row)

    # 合计行
    print("-" * (22 + 12 * (len(splits) + 1)))
    row = f"{'合计':<20}"
    grand_total = 0
    for split in splits:
        cnt = sum(len(v) for v in patient_list_by_split_cat[split].values())
        grand_total += cnt
        row += f" {cnt:>10}"
    row += f" {grand_total:>10}"
    print(row)

    print(f"\n{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(description="Bleeding Dataset 统计工具")
    parser.add_argument(
        "dataset_dir",
        type=str,
        nargs="?",
        default="data/Surge_Frames/Bleeding_Dataset_70_30",
        help="数据集目录路径",
    )
    parser.add_argument(
        "--json", "-j", action="store_true", help="输出 JSON 格式"
    )
    parser.add_argument(
        "--list", "-l", action="store_true", help="列出各 split 的病人编号详情"
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"错误: 目录不存在 {dataset_dir}")
        sys.exit(1)

    case_split_path = dataset_dir / "case_split.json"
    case_cache_path = dataset_dir / "_case_cache.json"
    train_meta_path = dataset_dir / "train_metadata.csv"
    test_meta_path = dataset_dir / "test_metadata.csv"
    val_meta_path = dataset_dir / "val_metadata.csv"

    # 检查必要文件
    if not case_split_path.exists():
        print(f"错误: 找不到 case_split.json")
        sys.exit(1)
    if not case_cache_path.exists():
        print(f"错误: 找不到 _case_cache.json")
        sys.exit(1)

    # 读取数据
    with open(case_split_path) as f:
        case_split = json.load(f)
    with open(case_cache_path) as f:
        case_cache = json.load(f)

    # 统计结果字典
    stats = {
        "dataset_dir": str(dataset_dir),
        "total_cases": len(case_split),
        "splits": {},
        "labels": {},
        "categories": {},
        "clips": {},
        "frames": {},
    }

    # [1] Case 级别划分统计
    split_counts = Counter(case_split.values())
    splits = sorted(split_counts.keys())

    # [2] 按原始类别统计
    split_by_cat = {s: Counter() for s in splits}
    for case_name, split in case_split.items():
        cat = categorize_case(case_name)
        split_by_cat[split][cat] += 1

    # [3] 标签分布 (case 级别)
    case_labels_by_split = {s: Counter() for s in splits}
    for case_name, split in case_split.items():
        if case_name in case_cache:
            label = case_cache[case_name]["label"]
            case_labels_by_split[split][label] += 1

    # [4] Clip 级别统计
    clip_counts_by_split = {s: 0 for s in splits}
    clip_labels_by_split = {s: Counter() for s in splits}
    for case_name, split in case_split.items():
        if case_name in case_cache:
            info = case_cache[case_name]
            n_clips = len(info["clips"])
            clip_counts_by_split[split] += n_clips
            clip_labels_by_split[split][info["label"]] += n_clips

    # [5] 各原始类别 clip 数量
    cat_clips = Counter()
    for case_name, info in case_cache.items():
        cat = categorize_case(case_name)
        cat_clips[cat] += len(info["clips"])

    # [6] 帧级别统计
    frame_stats = {}
    for split in splits:
        meta_path = dataset_dir / f"{split}_metadata.csv"
        if meta_path.exists():
            labels, cases = count_frames_by_label(str(meta_path))
            frame_stats[split] = {
                "total": sum(labels.values()),
                "labels": dict(labels),
                "cases": len(cases),
            }

    # 填充 stats 字典
    total_clips = sum(clip_counts_by_split.values())
    total_frames = sum(fs["total"] for fs in frame_stats.values())

    for split in splits:
        stats["splits"][split] = {
            "cases": split_counts[split],
            "cases_pct": round(100 * split_counts[split] / len(case_split), 1),
            "clips": clip_counts_by_split[split],
            "clips_pct": round(100 * clip_counts_by_split[split] / total_clips, 1) if total_clips > 0 else 0,
        }
        if split in frame_stats:
            stats["splits"][split]["frames"] = frame_stats[split]["total"]
            stats["splits"][split]["frames_pct"] = round(
                100 * frame_stats[split]["total"] / total_frames, 1
            ) if total_frames > 0 else 0

    stats["labels"] = {
        "bleeding_cases": sum(case_labels_by_split[s][1] for s in splits),
        "non_bleeding_cases": sum(case_labels_by_split[s][0] for s in splits),
        "bleeding_clips": sum(clip_labels_by_split[s][1] for s in splits),
        "non_bleeding_clips": sum(clip_labels_by_split[s][0] for s in splits),
    }

    stats["categories"] = dict(cat_clips)
    stats["clips"]["total"] = total_clips
    stats["frames"]["total"] = total_frames
    stats["frames"]["duration_minutes"] = round(total_frames / 60, 1)
    stats["frames"]["duration_hours"] = round(total_frames / 3600, 2)

    # 构建病人列表数据
    patient_list_by_split_cat = {s: {} for s in splits}
    for case_name, split in case_split.items():
        cat = categorize_case(case_name)
        if cat not in patient_list_by_split_cat[split]:
            patient_list_by_split_cat[split][cat] = []
        # 提取病人编号
        if case_name.startswith("Bleeding_"):
            pid = case_name.replace("Bleeding_", "")
        elif case_name.startswith("NonBleeding_"):
            pid = case_name.replace("NonBleeding_", "")
        elif case_name.startswith("NNB_"):
            pid = case_name.replace("NNB_", "")
        else:
            pid = case_name
        patient_list_by_split_cat[split][cat].append(pid)

    # 排序
    for split in splits:
        for cat in patient_list_by_split_cat[split]:
            patient_list_by_split_cat[split][cat].sort(key=lambda x: (len(x), x))

    # 添加到 stats
    stats["patient_list"] = {}
    for split in splits:
        stats["patient_list"][split] = {}
        for cat in sorted(patient_list_by_split_cat[split].keys()):
            stats["patient_list"][split][cat] = patient_list_by_split_cat[split][cat]

    # 输出
    if args.json:
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return

    # 列表模式：只显示病人划分详情
    if args.list:
        print_patient_list(splits, patient_list_by_split_cat, case_split)
        return

    # 文本格式输出
    print_section(f"Bleeding Dataset 统计: {dataset_dir}")

    print(f"\n[1] Case/Patient 级别划分")
    print(f"    {'划分':<10} {'Case数':>8} {'比例':>10}")
    print(f"    {'-'*30}")
    for split in splits:
        cnt = split_counts[split]
        pct = 100 * cnt / len(case_split)
        print(f"    {split:<10} {cnt:>8} {pct:>9.1f}%")
    print(f"    {'-'*30}")
    print(f"    {'总计':<10} {len(case_split):>8} {'100.0%':>10}")

    print(f"\n[2] 按原始数据类别")
    print(f"    {'类别':<20}", end="")
    for split in splits:
        print(f" {split:>8}", end="")
    print(f" {'合计':>8}")
    print(f"    {'-'*50}")
    all_cats = sorted(set(c for s in split_by_cat.values() for c in s.keys()))
    for cat in all_cats:
        print(f"    {cat:<20}", end="")
        row_total = 0
        for split in splits:
            cnt = split_by_cat[split].get(cat, 0)
            row_total += cnt
            print(f" {cnt:>8}", end="")
        print(f" {row_total:>8}")

    print(f"\n[3] 标签分布 (Case 级别)")
    print(f"    {'划分':<10} {'Bleeding':>12} {'Non-bleeding':>14}")
    print(f"    {'-'*40}")
    for split in splits:
        b = case_labels_by_split[split].get(1, 0)
        nb = case_labels_by_split[split].get(0, 0)
        print(f"    {split:<10} {b:>12} {nb:>14}")
    total_b = stats["labels"]["bleeding_cases"]
    total_nb = stats["labels"]["non_bleeding_cases"]
    print(f"    {'-'*40}")
    print(f"    {'总计':<10} {total_b:>12} ({100*total_b/(total_b+total_nb):.1f}%) {total_nb:>8} ({100*total_nb/(total_b+total_nb):.1f}%)")

    print(f"\n[4] Clip 级别统计")
    print(f"    {'划分':<10} {'Clips':>10} {'比例':>10} {'Bleeding':>12} {'Non-bleeding':>14}")
    print(f"    {'-'*60}")
    for split in splits:
        cnt = clip_counts_by_split[split]
        pct = 100 * cnt / total_clips if total_clips > 0 else 0
        b = clip_labels_by_split[split].get(1, 0)
        nb = clip_labels_by_split[split].get(0, 0)
        print(f"    {split:<10} {cnt:>10} {pct:>9.1f}% {b:>12} {nb:>14}")
    print(f"    {'-'*60}")
    print(f"    {'总计':<10} {total_clips:>10} {'100.0%':>10}")

    print(f"\n[5] 各原始类别 Clip 数量")
    for cat, cnt in sorted(cat_clips.items()):
        print(f"    {cat}: {cnt} clips")

    if frame_stats:
        print(f"\n[6] 帧级别统计 (1 FPS 抽帧)")
        print(f"    {'划分':<10} {'帧数':>10} {'比例':>10} {'Bleeding':>12} {'Non-bleeding':>14}")
        print(f"    {'-'*60}")
        for split in splits:
            if split in frame_stats:
                fs = frame_stats[split]
                pct = 100 * fs["total"] / total_frames if total_frames > 0 else 0
                b = fs["labels"].get(1, 0)
                nb = fs["labels"].get(0, 0)
                print(f"    {split:<10} {fs['total']:>10} {pct:>9.1f}% {b:>12} {nb:>14}")
        print(f"    {'-'*60}")
        print(f"    {'总计':<10} {total_frames:>10} {'100.0%':>10}")

        print(f"\n[7] 估算视频时长")
        print(f"    总帧数: {total_frames}")
        print(f"    时长: ~{total_frames/60:.1f} 分钟 ≈ {total_frames/3600:.2f} 小时")

    print(f"\n{'=' * 70}")
    print(f" 统计完成")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
