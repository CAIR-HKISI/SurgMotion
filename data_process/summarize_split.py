#!/usr/bin/env python3
"""
统计 case_split.json 的划分详情，按类别和分割集输出病人列表及视频数。

用法:
    python data_process/summarize_split.py
    python data_process/summarize_split.py --split_json data/Surge_Frames/Bleeding_V2/case_split.json
"""
import json
import re
import argparse
from collections import defaultdict


CATEGORY_ORDER = ["Bleeding", "NonBleeding", "new_Nonbleeding"]
SPLIT_ORDER = ["train", "val", "test"]

CATEGORY_DISPLAY = {
    "Bleeding": "Bleeding",
    "NonBleeding": "NonBleeding",
    "new_Nonbleeding": "new_Nonbleeding (NNB)",
}


def parse_case_name(case_name: str):
    """从 case_name 提取 category, patient_id(数字), case 序号."""
    if case_name.startswith("Bleeding_"):
        m = re.match(r"Bleeding_(\d+)_case_(\d+)", case_name)
        if m:
            return "Bleeding", int(m.group(1)), int(m.group(2))
    elif case_name.startswith("NonBleeding_"):
        m = re.match(r"NonBleeding_(\d+)_case_(\d+)", case_name)
        if m:
            return "NonBleeding", int(m.group(1)), int(m.group(2))
    elif case_name.startswith("NNB_"):
        m = re.match(r"NNB_(\d+)_case_(\d+)", case_name)
        if m:
            return "new_Nonbleeding", int(m.group(1)), int(m.group(2))
    return None, None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split_json", default="data/Surge_Frames/Bleeding_V2/case_split.json",
    )
    args = parser.parse_args()

    with open(args.split_json) as f:
        case_split = json.load(f)

    # {split: {category: {patient_id: video_count}}}
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for case_name, split in case_split.items():
        cat, pid, _ = parse_case_name(case_name)
        if cat is None:
            continue
        stats[split][cat][pid] += 1

    active_splits = [s for s in SPLIT_ORDER if s in stats]
    total_patients_all = 0
    total_videos_all = 0

    col_width = max(46, 46)
    separator = "│"

    # 打印表头
    header_parts = []
    for sp in active_splits:
        total_p = sum(len(stats[sp][c]) for c in CATEGORY_ORDER if c in stats[sp])
        total_v = sum(sum(stats[sp][c].values()) for c in CATEGORY_ORDER if c in stats[sp])
        ratio = total_v / len(case_split) * 100
        label = f"{sp} ({ratio:.0f}%): {total_p} 病人, {total_v} 视频"
        header_parts.append(label)
        total_patients_all += total_p
        total_videos_all += total_v

    print("=" * (col_width * len(active_splits) + len(active_splits) - 1))
    print(f"  数据集划分统计: {args.split_json}")
    print(f"  总计: {total_patients_all} 病人, {total_videos_all} 视频, "
          f"{len(active_splits)} 个分割集")
    print("=" * (col_width * len(active_splits) + len(active_splits) - 1))
    print()

    header_line = f" {separator} ".join(
        f"  {h:<{col_width - 4}}" for h in header_parts
    )
    print(header_line)
    print("─" * len(header_line))

    for cat in CATEGORY_ORDER:
        has_cat = any(cat in stats[sp] for sp in active_splits)
        if not has_cat:
            continue

        parts = []
        for sp in active_splits:
            patients = stats[sp].get(cat, {})
            n_patients = len(patients)
            n_videos = sum(patients.values())
            display_name = CATEGORY_DISPLAY.get(cat, cat)
            cat_header = f"  {display_name} ({n_patients} 病人, {n_videos} 视频)"
            parts.append((cat_header, patients))

        # 类别标题行
        title_line = f" {separator} ".join(
            p[0].ljust(col_width - 2) for p in parts
        )
        print(title_line)

        # 病人 ID 列表（带每个病人的视频数）
        formatted_parts = []
        for _, patients in parts:
            if not patients:
                formatted_parts.append([])
                continue
            sorted_pids = sorted(patients.keys())
            items = []
            for pid in sorted_pids:
                vc = patients[pid]
                if vc > 1:
                    items.append(f"{pid}({vc})")
                else:
                    items.append(str(pid))
            formatted_parts.append(items)

        max_lines = 0
        wrapped_parts = []
        for items in formatted_parts:
            lines = []
            current_line = "    "
            for i, item in enumerate(items):
                addition = item + (", " if i < len(items) - 1 else "")
                if len(current_line) + len(addition) > col_width - 4:
                    lines.append(current_line.rstrip(", "))
                    current_line = "    " + addition
                else:
                    current_line += addition
            if current_line.strip():
                lines.append(current_line.rstrip(", "))
            wrapped_parts.append(lines)
            max_lines = max(max_lines, len(lines))

        for line_idx in range(max_lines):
            row_parts = []
            for wp in wrapped_parts:
                if line_idx < len(wp):
                    row_parts.append(wp[line_idx].ljust(col_width - 2))
                else:
                    row_parts.append(" " * (col_width - 2))
            print(f" {separator} ".join(row_parts))

        print()

    # 汇总表格
    print("─" * (col_width * len(active_splits) + len(active_splits) - 1))
    print("\n汇总统计:")
    print(f"{'类别':<24}", end="")
    for sp in active_splits:
        print(f"  {sp:>8}(病人/视频)", end="")
    print(f"  {'总计':>8}")
    print("-" * (24 + 22 * (len(active_splits) + 1)))

    for cat in CATEGORY_ORDER:
        has_cat = any(cat in stats[sp] for sp in active_splits)
        if not has_cat:
            continue
        display = CATEGORY_DISPLAY.get(cat, cat)
        print(f"{display:<24}", end="")
        cat_total_p, cat_total_v = 0, 0
        for sp in active_splits:
            patients = stats[sp].get(cat, {})
            np_ = len(patients)
            nv_ = sum(patients.values())
            cat_total_p += np_
            cat_total_v += nv_
            print(f"  {np_:>5} / {nv_:<6}", end="")
        print(f"  {cat_total_p:>5} / {cat_total_v:<6}")

    print("-" * (24 + 22 * (len(active_splits) + 1)))
    print(f"{'合计':<24}", end="")
    for sp in active_splits:
        sp_p = sum(len(stats[sp][c]) for c in CATEGORY_ORDER if c in stats[sp])
        sp_v = sum(sum(stats[sp][c].values()) for c in CATEGORY_ORDER if c in stats[sp])
        print(f"  {sp_p:>5} / {sp_v:<6}", end="")
    print(f"  {total_patients_all:>5} / {total_videos_all:<6}")


if __name__ == "__main__":
    main()
