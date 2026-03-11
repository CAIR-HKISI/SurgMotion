#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

DATASETS = {
    "phase": {
        "train": REPO_ROOT
        / "data/Surge_Frames/Cholec80/clips_64f_anticipation/train_dense_64f_detailed.csv",
        "val": REPO_ROOT
        / "data/Surge_Frames/Cholec80/clips_64f_anticipation/val_dense_64f_detailed.csv",
    },
    "instrument": {
        "train": REPO_ROOT
        / "data/Surge_Frames/Cholec80/clips_64f_instrument_anticipation/train_dense_64f_detailed.csv",
        "val": REPO_ROOT
        / "data/Surge_Frames/Cholec80/clips_64f_instrument_anticipation/val_dense_64f_detailed.csv",
    },
}

# 直接在这里填写想采样的视频号；例如 [1, 3, 5, 10] 会转换成
# ["video01", "video03", "video05", "video10"]。
DEFAULT_TRAIN_VIDEO_IDS = list(range(1, 21))
DEFAULT_VAL_VIDEO_IDS = list(range(41, 46))


def video_ids_to_names(video_ids: list[int]) -> list[str]:
    invalid_ids = [video_id for video_id in video_ids if video_id <= 0]
    if invalid_ids:
        raise ValueError(f"视频号必须为正整数，非法值: {invalid_ids}")

    deduped = []
    seen = set()
    for video_id in video_ids:
        video_name = f"video{video_id:02d}"
        if video_name not in seen:
            seen.add(video_name)
            deduped.append(video_name)
    return deduped


def parse_video_list(value: str | None, default: list[str]) -> list[str]:
    if value is None:
        return list(default)

    parsed = [item.strip() for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("视频列表为空，请传入逗号分隔的 video 名称。")

    deduped = []
    seen = set()
    for name in parsed:
        if name not in seen:
            seen.add(name)
            deduped.append(name)
    return deduped


def load_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]], list[str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames:
        raise ValueError(f"CSV 缺少表头: {csv_path}")

    ordered_videos = []
    seen = set()
    for row in rows:
        video_name = row["video_name"]
        if video_name not in seen:
            seen.add(video_name)
            ordered_videos.append(video_name)

    return fieldnames, rows, ordered_videos


def write_subset_csv(
    source_csv: Path,
    target_csv: Path,
    requested_videos: list[str],
) -> tuple[int, list[str]]:
    fieldnames, rows, ordered_videos = load_rows(source_csv)
    missing = [name for name in requested_videos if name not in ordered_videos]
    if missing:
        raise ValueError(f"{source_csv} 中不存在这些视频: {missing}")

    requested_set = set(requested_videos)
    subset_rows = [row for row in rows if row["video_name"] in requested_set]

    if "Index" in fieldnames:
        for idx, row in enumerate(subset_rows):
            row["Index"] = str(idx)

    target_csv.parent.mkdir(parents=True, exist_ok=True)
    with target_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(subset_rows)

    ordered_requested = [name for name in ordered_videos if name in requested_set]
    return len(subset_rows), ordered_requested


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成 Cholec80 phase/instrument anticipation 的小规模 CSV。"
    )
    parser.add_argument(
        "--train-videos",
        type=str,
        default=None,
        help="逗号分隔的训练视频列表，例如 video01,video02,video03",
    )
    parser.add_argument(
        "--val-videos",
        type=str,
        default=None,
        help="逗号分隔的测试视频列表，例如 video41,video42,video43",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="small",
        help="输出 CSV 文件后缀，默认生成 *_small.csv",
    )
    args = parser.parse_args()

    default_train_videos = video_ids_to_names(DEFAULT_TRAIN_VIDEO_IDS)
    default_val_videos = video_ids_to_names(DEFAULT_VAL_VIDEO_IDS)
    train_videos = parse_video_list(args.train_videos, default_train_videos)
    val_videos = parse_video_list(args.val_videos, default_val_videos)

    print(f"Train videos ({len(train_videos)}): {train_videos}")
    print(f"Val videos   ({len(val_videos)}): {val_videos}")

    for dataset_name, paths in DATASETS.items():
        train_target = paths["train"].with_name(
            paths["train"].stem + f"_{args.suffix}" + paths["train"].suffix
        )
        val_target = paths["val"].with_name(
            paths["val"].stem + f"_{args.suffix}" + paths["val"].suffix
        )

        train_clip_count, ordered_train = write_subset_csv(
            paths["train"], train_target, train_videos
        )
        val_clip_count, ordered_val = write_subset_csv(
            paths["val"], val_target, val_videos
        )

        print(f"[{dataset_name}] train -> {train_target}")
        print(f"[{dataset_name}]   clips: {train_clip_count}")
        print(f"[{dataset_name}]   videos: {ordered_train}")
        print(f"[{dataset_name}] val   -> {val_target}")
        print(f"[{dataset_name}]   clips: {val_clip_count}")
        print(f"[{dataset_name}]   videos: {ordered_val}")


if __name__ == "__main__":
    main()
