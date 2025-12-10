"""
基于 JIGSAWS 视频抽帧、生成帧列表与按 4-fold 划分的 CSV。

配置来源：
- 视频：data/Landscopy/JIGSAWS_CVPR2021/data/JIGSAWS/video_encoded
- 标签：data/Landscopy/JIGSAWS_CVPR2021/data/JIGSAWS/label (FT-score-*.npy)
- 划分：data/Landscopy/JIGSAWS_CVPR2021/Towards-Unified-Surgical-Skill-Assessment/metas/JIGSAWS_split_4_fold.npy

输出：
- 帧：data/Surge_Frames/JIGSAWS_v1/frames_{fps}/{video_id}/%06d.jpg
- 帧列表：data/Surge_Frames/JIGSAWS_v1/clip_{fps}f.txt  每行：<video_id> <abs_frames_dir>
- CSV（按折拆分 train/test）：data/Surge_Frames/JIGSAWS_v1/{split}_fold{fold}_{fps}f.csv
  列：Index, case_id, clip_path, label, label_name
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


# ===== 基本路径与配置 =====
VIDEO_DIR = Path("data/Landscopy/JIGSAWS_CVPR2021/data/JIGSAWS/video_encoded")
LABEL_DIR = Path("data/Landscopy/JIGSAWS_CVPR2021/data/JIGSAWS/label")
SPLIT_PATH = Path(
    "data/Landscopy/JIGSAWS_CVPR2021/Towards-Unified-Surgical-Skill-Assessment/metas/JIGSAWS_split_4_fold.npy"
)

FRAME_ROOT = Path("data/Surge_Frames/JIGSAWS_v1")
FPS_LIST = [1,5,15,20]

LABEL_KEY = "GRS"
LABEL_RANGE = (6, 30)  # 归一化区间


# ===== 工具函数 =====
def load_splits() -> Dict[str, List[List[str]]]:
    return np.load(SPLIT_PATH, allow_pickle=True).item()


def build_video_map() -> Dict[str, List[str]]:
    """
    将不带 capture 后缀的基础名映射到实际存在的 <video_id>（含 captureX）。
    例：Knot_Tying_B001 -> [Knot_Tying_B001_capture1, Knot_Tying_B001_capture2]
    """
    mapping: Dict[str, List[str]] = {}
    for mp4 in VIDEO_DIR.glob("*.mp4"):
        base = mp4.stem  # 带 capture 后缀
        if "_capture" in base:
            root = base.split("_capture")[0]
        else:
            root = base
        mapping.setdefault(root, []).append(base)
    return mapping


def get_all_videos(splits: Dict[str, List[List[str]]]) -> List[str]:
    vids: List[str] = []
    for fold_lists in splits.values():
        for fold in fold_lists:
            vids.extend(fold)
    return sorted(set(vids))


def expand_split_ids(split_ids: Iterable[str], video_map: Dict[str, List[str]]) -> List[str]:
    """将无 capture 的名字展开为实际存在的 video_id 列表。"""
    expanded: List[str] = []
    for name in split_ids:
        matches = video_map.get(name, [])
        if not matches:
            print(f"[WARN] no matched video for '{name}' (expecting capture*)")
        expanded.extend(matches)
    return expanded


def expand_splits(
    splits: Dict[str, List[List[str]]], video_map: Dict[str, List[str]]
) -> Dict[str, List[List[str]]]:
    """对每个折的列表做 capture 展开。"""
    new_splits: Dict[str, List[List[str]]] = {}
    for task, fold_lists in splits.items():
        new_fold_lists: List[List[str]] = []
        for fold in fold_lists:
            new_fold_lists.append(expand_split_ids(fold, video_map))
        new_splits[task] = new_fold_lists
    return new_splits


def parse_case_id(video_id: str) -> int:
    """从 video_id 中提取主体编号，如 Knot_Tying_B003_capture2 -> 3"""
    parts = video_id.split("_")
    if len(parts) < 2:
        return -1
    subject = parts[-2]
    m = re.search(r"(\\d+)", subject)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return -1
    return -1


def load_label(video_id: str) -> float | None:
    npy = LABEL_DIR / f"FT-score-{video_id}.npy"
    if not npy.exists():
        print(f"[WARN] label missing: {npy}")
        return None
    obj = np.load(npy, allow_pickle=True).item()
    if LABEL_KEY not in obj:
        print(f"[WARN] label key '{LABEL_KEY}' missing in {npy.name}")
        return None
    raw = float(obj[LABEL_KEY])
    lo, hi = LABEL_RANGE
    norm = (raw - lo) / (hi - lo)
    return norm


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def extract_frames(video_id: str, fps: int) -> None:
    src = VIDEO_DIR / f"{video_id}.mp4"
    if not src.exists():
        print(f"[WARN] video missing: {src}")
        return
    dst_dir = FRAME_ROOT / f"frames_{fps}" / video_id
    # 若已存在帧则跳过
    if dst_dir.exists() and any(dst_dir.iterdir()):
        return
    ensure_dir(dst_dir)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",
        str(dst_dir / "%06d.jpg"),
    ]
    print(f"[INFO] extracting {video_id} @ {fps} fps")
    subprocess.run(cmd, check=True)


def write_clip_txt(video_ids: Iterable[str], fps: int) -> None:
    lines = []
    for vid in video_ids:
        frames_dir = (FRAME_ROOT / f"frames_{fps}" / vid).resolve()
        if frames_dir.exists():
            lines.append(f"{vid} {frames_dir}")
    out_path = FRAME_ROOT / f"clip_{fps}f.txt"
    ensure_dir(out_path.parent)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] clip list saved: {out_path} ({len(lines)} lines)")


def build_split_csv(
    fps: int,
    fold_idx: int,
    split_name: str,
    video_ids: Iterable[str],
    start_index: int = 0,
) -> Tuple[pd.DataFrame, int]:
    rows = []
    idx = start_index
    for vid in video_ids:
        frames_dir = (FRAME_ROOT / f"frames_{fps}" / vid).resolve()
        if not frames_dir.exists():
            print(f"[WARN] frames not found for {vid} @ {fps}fps, skip")
            continue
        label = load_label(vid)
        if label is None:
            continue
        case_id = parse_case_id(vid)
        rows.append(
            {
                "Index": idx,
                "case_id": case_id,
                "clip_path": str(frames_dir),
                "label": label,
                "label_name": LABEL_KEY,
            }
        )
        idx += 1

    df = pd.DataFrame(rows)
    out_csv = FRAME_ROOT / f"{split_name}_fold{fold_idx}_{fps}f.csv"
    df.to_csv(out_csv, index=False)
    print(f"[INFO] saved {len(df)} rows -> {out_csv}")
    return df, idx


def main() -> None:
    raw_splits = load_splits()
    video_map = build_video_map()
    splits = expand_splits(raw_splits, video_map)
    all_videos = get_all_videos(splits)

    # 1) 抽帧
    for fps in FPS_LIST:
        for vid in all_videos:
            extract_frames(vid, fps)

    # 2) clip_{fps}f.txt
    for fps in FPS_LIST:
        write_clip_txt(all_videos, fps)

    # 3) 按折生成 CSV（train/test），每个 fps 一套
    for fps in FPS_LIST:
        for fold_idx in range(4):
            train_videos: List[str] = []
            test_videos: List[str] = []
            for task, task_folds in splits.items():
                if fold_idx >= len(task_folds):
                    continue
                test_videos.extend(task_folds[fold_idx])
                for j, fold in enumerate(task_folds):
                    if j != fold_idx:
                        train_videos.extend(fold)
            # 去重保持稳定顺序
            train_videos = sorted(set(train_videos))
            test_videos = sorted(set(test_videos))
            build_split_csv(fps, fold_idx, "train", train_videos, start_index=0)
            build_split_csv(fps, fold_idx, "test", test_videos, start_index=0)


if __name__ == "__main__":
    main()

