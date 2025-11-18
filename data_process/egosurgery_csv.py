import os
import pandas as pd
from tqdm import tqdm

# === 路径配置 ===
ANNOT_PATH = "data/Open_surgery/EgoSurgery/annotations/phase"
FRAME_ROOT = "data/Surge_Frames/EgoSurgery/frames"
OUT_DIR = "data/Surge_Frames/EgoSurgery"

# === Phase 映射表 ===
phase2id = {
    'disinfection': 0,
    'design': 1,
    'anesthesia': 2,
    'incision': 3,
    'dissection': 4,
    'hemostasis': 5,
    'irrigation': 6,
    'closure': 7,
    'dressing': 8
}

# === 视频划分 ===
TRAIN_VIDEOS =  [1, 2, 3, 4, 8, 9, 11, 13, 14, 15, 17, 20, 21]
VAL_VIDEOS = [5, 19]
TEST_VIDEOS = [6,7,10,12,18]

def get_split(case_id):
    """根据视频编号确定所属数据集划分"""
    if case_id in TRAIN_VIDEOS:
        return "train"
    elif case_id in VAL_VIDEOS:
        return "val"
    elif case_id in TEST_VIDEOS:
        return "test"
    else:
        return None

def generate_egosurgery_csv():
    all_data = []
    missing_report = []  # 用于记录缺失帧信息
    global_idx = 0

    os.makedirs(OUT_DIR, exist_ok=True)

    # 遍历每个标注文件
    for annot_file in tqdm(sorted(os.listdir(ANNOT_PATH)), desc="Processing EgoSurgery annotations"):
        if not annot_file.endswith(".csv"):
            continue

        # 文件名格式: "01_1.csv"
        case_name = annot_file.replace(".csv", "")  # e.g. 01_1
        try:
            video_id, view_id = case_name.split("_")
            case_id = int(video_id)
            case_view = int(view_id)
        except ValueError:
            print(f"⚠️ Skipping file with unexpected name: {annot_file}")
            continue

        split = get_split(case_id)
        if split is None:
            print(f"⚠️ Skipped {annot_file} (case_id={case_id} not in defined splits)")
            continue

        # 读取标注
        df_phase = pd.read_csv(os.path.join(ANNOT_PATH, annot_file))
        if "Frame" not in df_phase.columns or "Phase" not in df_phase.columns:
            print(f"⚠️ Missing columns in {annot_file}")
            continue

        total_frames = len(df_phase)
        missing_count = 0

        for _, row in df_phase.iterrows():
            frame_name = f"{row['Frame']}.jpg"
            frame_path = os.path.join(FRAME_ROOT, video_id, frame_name)

            if not os.path.exists(frame_path):
                missing_report.append({
                    "Video": video_id,
                    "View": view_id,
                    "Missing_Frame": frame_name,
                    "Annotation_File": annot_file
                })
                missing_count += 1
                continue

            phase_name = str(row["Phase"]).strip().lower()
            phase_gt = phase2id.get(phase_name, -1)

            data_item = {
                'index': global_idx,
                'DataName': 'EgoSurgery',
                'Year': 2023,
                'Case_Name': f"video{video_id}_view{view_id}",
                'Case_ID': case_id,
                'View_ID': case_view,
                'Frame_Path': frame_path,
                'Phase_GT': phase_gt,
                'Phase_Name': phase_name,
                'Split': split
            }
            all_data.append(data_item)
            global_idx += 1

        if missing_count > 0:
            print(f"⚠️ {annot_file}: {missing_count}/{total_frames} frames missing")
        else:
            print(f"✅ {annot_file}: all {total_frames} frames found")

    # === 输出 CSV（train/val/test） ===
    if all_data:
        df = pd.DataFrame(all_data)
        print(f"\n✅ Total valid frames: {len(df)}")

        for split_name in ["train", "val", "test"]:
            df_split = df[df["Split"] == split_name]
            if not df_split.empty:
                out_csv = os.path.join(OUT_DIR, f"{split_name}_metadata.csv")
                df_split.to_csv(out_csv, index=False)
                print(f"💾 Saved {len(df_split)} rows to {out_csv}")
            else:
                print(f"⚠️ No data for split: {split_name}")
    else:
        print("⚠️ No valid frame data found, skipping CSV save.")

    # === 输出缺失帧报告 ===
    if missing_report:
        df_missing = pd.DataFrame(missing_report)
        out_miss_csv = os.path.join(OUT_DIR, "missing_frames_report.csv")
        df_missing.to_csv(out_miss_csv, index=False)
        print(f"\n❌ Missing frames detected! Report saved to {out_miss_csv}")
        print(df_missing.groupby("Annotation_File").size())
    else:
        print("\n✅ All annotated frames have corresponding images!")

if __name__ == "__main__":
    generate_egosurgery_csv()
