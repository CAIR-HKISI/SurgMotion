import os
import pandas as pd
from tqdm import tqdm

# ========================= #
# 🔧 配置区域
# ========================= #
FRAME_PATH = "data/Surge_Frames/PolypDiag/frames"
SPLIT_DIR = "data/GI_Videos/PolypDiag/splits"
OUT_DIR = "data/Surge_Frames/PolypDiag"

# 输出 CSV 文件
TRAIN_CSV = os.path.join(OUT_DIR, "train_metadata.csv")
TEST_CSV = os.path.join(OUT_DIR, "test_metadata.csv")

DATA_NAME = "PolypDiag"
YEAR = 2025


# ========================= #
# 🧩 加载 splits
# ========================= #
def load_split_file(split_path):
    """读取train/val.txt，返回 {video_name: label_int}"""
    mapping = {}
    if not os.path.exists(split_path):
        print(f"❌ Error: split file does not exist -> {split_path}")
        return mapping

    with open(split_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "," not in line:
                continue
            name, label = line.split(",")
            mapping[name.replace(".mp4", "")] = int(label)

    print(f"✅ Loaded {len(mapping)} items from {split_path}")
    return mapping


# ========================= #
# 🧮 生成 metadata.csv
# ========================= #
def generate_metadata_file(video_label_map, split_name, out_csv):
    all_data = []
    global_idx = 0
    missing_frames = 0
    missing_videos = 0

    for video_name, label in tqdm(video_label_map.items(), desc=f"Processing {split_name} set"):
        case_name = video_name
        case_id = len(all_data) + 1
        video_dir = os.path.join(FRAME_PATH, case_name)

        if not os.path.exists(video_dir):
            print(f"⚠️ Warning: Frame folder NOT found: {video_dir}")
            missing_videos += 1
            continue

        frame_files = sorted(
            [f for f in os.listdir(video_dir) if f.lower().endswith((".jpg", ".png"))]
        )

        if not frame_files:
            print(f"⚠️ No frames found in {video_dir}")
            continue

        label_name = "Abnormal" if label == 1 else "Normal"

        for frame_file in frame_files:
            frame_path = os.path.join(video_dir, frame_file)

            # ✅ 检查帧文件是否存在
            if not os.path.exists(frame_path):
                missing_frames += 1
                continue

            data_item = {
                "index": global_idx,
                "DataName": DATA_NAME,
                "Year": YEAR,
                "Case_Name": case_name,
                "Case_ID": case_id,
                "Frame_Path": frame_path,
                "Phase_GT": label,       # 标签 (0/1)
                "Phase_Name": label_name,
                "Split": split_name
            }
            all_data.append(data_item)
            global_idx += 1

    # 导出 CSV
    if all_data:
        df = pd.DataFrame(all_data)
        os.makedirs(OUT_DIR, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"💾 Saved {len(df)} valid frames to {out_csv}")
    else:
        print(f"⚠️ No valid frames processed for {split_name} set.")

    print(f"📊 Summary for {split_name}:")
    print(f"   - Missing video dirs: {missing_videos}")
    print(f"   - Missing frames: {missing_frames}")
    print(f"   - Saved samples: {len(all_data)}")


# ========================= #
# 🚀 主程序入口
# ========================= #
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    train_map = load_split_file(os.path.join(SPLIT_DIR, "train.txt"))
    test_map = load_split_file(os.path.join(SPLIT_DIR, "val.txt"))

    generate_metadata_file(train_map, "train", TRAIN_CSV)
    generate_metadata_file(test_map, "test", TEST_CSV)

    print("\n🎉 PolypDiag train/test metadata CSVs generated successfully!")