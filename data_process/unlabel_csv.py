import os
import pandas as pd
from tqdm import tqdm
import argparse


def get_all_videos(frames_dir: str):
    """
    递归获取所有视频文件夹路径。
    条件：视频文件夹下至少包含一个 jpg/png 文件。
    """
    video_dirs = []
    for root, dirs, files in os.walk(frames_dir):
        has_image = any(f.lower().endswith((".jpg", ".png")) for f in files)
        if has_image:
            video_dirs.append(root)
    return sorted(video_dirs)


def extract_number(s: str) -> int:
    """从字符串中提取第一个数字，否则返回 -1"""
    import re
    matches = re.findall(r"\d+", s)
    return int(matches[0]) if matches else -1


def generate_unlabeled_csv(root_dir: str, data_name: str):
    """
    从无标注视频帧目录递归生成 unlabeled_metadata.csv
    
    输出 CSV 字段：
        index, DataName, Year, Case_Name, Case_ID, Frame_Path, Phase_GT, Phase_Name, Split
    """
    frames_dir = os.path.join(root_dir, "frames")
    out_csv = os.path.join(root_dir, "unlabeled_metadata.csv")

    if not os.path.exists(frames_dir):
        print(f"❌ Error: frames directory not found: {frames_dir}")
        return

    # 递归查找含图像的目录
    video_dirs = get_all_videos(frames_dir)
    if not video_dirs:
        print(f"⚠️ No video folders found in {frames_dir}")
        return

    print(f"📂 Found {len(video_dirs)} video folders (recursively) in {frames_dir}")

    all_data = []
    global_idx = 0

    for video_dir in tqdm(video_dirs, desc="Processing videos"):
        # e.g. frames/video01/sub1 → "video01_sub1"
        relative_path = os.path.relpath(video_dir, frames_dir)
        case_name = relative_path.replace(os.sep, "_")
        case_id = extract_number(case_name)

        # 获取目录中所有帧（递归）
        frame_files = []
        for root, _, files in os.walk(video_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".png")):
                    frame_files.append(os.path.join(root, f))

        # 按文件名数字排序：00001.jpg < 00002.jpg
        frame_files = sorted(frame_files, key=lambda x: extract_number(os.path.basename(x)))

        if not frame_files:
            print(f"⚠️ Warning: No frames found in {video_dir}")
            continue

        for frame_path in frame_files:
            item = {
                "Index": global_idx,
                "DataName": data_name,
                "Year": 2024,
                "Case_Name": case_name,
                "Case_ID": case_id,
                "Frame_Path": frame_path,
                "Phase_GT": -1,
                "Phase_Name": "none",
                "Split": "unlabeled",
            }
            all_data.append(item)
            global_idx += 1

    if not all_data:
        print("⚠️ No frames found — CSV not generated.")
        return

    df = pd.DataFrame(all_data)
    df.to_csv(out_csv, index=False)
    print(f"✅ Saved {len(df)} frames to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursively generate unlabeled metadata CSV for frame-only dataset.")
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Path to the root directory (e.g., data/Surge_Frames/AutoLaparo)")
    parser.add_argument("--data_name", type=str, required=True,
                        help="Name of the dataset (e.g., AutoLaparo)")
    args = parser.parse_args()

    generate_unlabeled_csv(args.root_dir, args.data_name)

