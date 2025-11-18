import os
import pickle
import pandas as pd
from tqdm import tqdm

# ======== 路径配置 ========
FRAME_PATH = "data/Surge_Frames/PmLR50/frames"
OUT_DIR = "data/Surge_Frames/PmLR50"
os.makedirs(OUT_DIR, exist_ok=True)

LABEL_PATHS = {
    "train": "data/Landscopy/PmLR50/PmLR50/labels/train/1fpstrain.pickle",
    "test":   "data/Landscopy/PmLR50/PmLR50/labels/infer/1fpsinfer.pickle",
    "val":  "data/Landscopy/PmLR50/PmLR50/labels/test/1fpstest.pickle",
}

# ======== 阶段编号映射 ========
phase2name = {
    0: "Preparation stage",
    1: "Knotting of the Foley catheter",
    2: "Procedure of the liver resection",
    3: "Release of the Foley catheter",
    4: "Postprocessing stage"
}

def generate_pmlr_csv():
    all_data = []
    missing_frames = []
    global_idx = 0

    for split, label_path in LABEL_PATHS.items():
        print(f"\n📂 Processing split: {split} ({label_path})")

        if not os.path.exists(label_path):
            print(f"⚠️ Label file {label_path} not found, skipping.")
            continue

        # 加载 pickle 文件
        with open(label_path, "rb") as f:
            data = pickle.load(f)

        # 遍历每个视频
        for vid, entries in tqdm(data.items(), desc=f"Processing {split} videos"):
            try:
                case_id = int(vid)
            except Exception:
                print(f"⚠️ Invalid video id: {vid}")
                continue

            video_dir = os.path.join(FRAME_PATH, f"{case_id:02d}")
            if not os.path.exists(video_dir):
                print(f"⚠️ Missing frame directory: {video_dir}")
                continue

            for entry in entries:
                frame_id = int(entry["frame_id"])
                frame_file = f"{frame_id:08d}.jpg"
                frame_path = os.path.join(video_dir, frame_file)

                if not os.path.exists(frame_path):
                    missing_frames.append({
                        "Split": split,
                        "Video": case_id,
                        "Missing_Frame": frame_file
                    })
                    continue

                phase_gt = entry.get("phase_gt", -1)
                phase_name = phase2name.get(phase_gt, "Unknown")

                data_item = {
                    "Index": global_idx,
                    "DataName": "PmLR50",
                    "Year": 2023,
                    "Case_Name": f"video{case_id:02d}",
                    "Case_ID": case_id,
                    "Frame_ID": frame_id,
                    "Frame_Path": frame_path,
                    "Phase_GT": phase_gt,
                    "Phase_Name": phase_name,
                    "Split": split
                }

                all_data.append(data_item)
                global_idx += 1

    # === 汇总结果 ===
    if not all_data:
        print("❌ No valid data found.")
        return

    df = pd.DataFrame(all_data)
    print(f"\n✅ Total processed frames: {len(df)}")

    # === 保存总 metadata ===
    merged_csv = os.path.join(OUT_DIR, "metadata.csv")
    df.to_csv(merged_csv, index=False)
    print(f"💾 Saved combined metadata to {merged_csv}")

    # === 保存各 split 的 metadata ===
    for split in ["train", "val", "test"]:
        df_split = df[df["Split"] == split]
        if not df_split.empty:
            out_csv = os.path.join(OUT_DIR, f"{split}_metadata.csv")
            df_split.to_csv(out_csv, index=False)
            print(f"💾 Saved {len(df_split)} frames to {out_csv}")
        else:
            print(f"⚠️ No data found for split: {split}")

    # === 缺帧报告 ===
    if missing_frames:
        df_missing = pd.DataFrame(missing_frames)
        miss_csv = os.path.join(OUT_DIR, "missing_frames_report.csv")
        df_missing.to_csv(miss_csv, index=False)
        print(f"\n❌ Missing frames detected! Saved report to {miss_csv}")
        print(df_missing.groupby("Video").size())
    else:
        print("\n✅ All annotated frames exist!")

if __name__ == "__main__":
    generate_pmlr_csv()

