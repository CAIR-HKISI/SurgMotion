import os
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# === 路径配置 ===
FRAME_ROOT = "data/Surge_Frames/AIxsuture/frames"
META_PATH = "data/Landscopy/AIxsuture/OSATS.xlsx"
OUT_DIR = "data/Surge_Frames/AIxsuture"

YEAR = 2025
DATANAME = "AIxsuture"
PHASE_NAME = "Suturing"

# === 函数定义 ===

def load_video_scores(xlsx_path):
    """从 OSATS.xlsx 读取并计算每个视频的平均 GLOBA_RATING_SCORE"""
    df = pd.read_excel(xlsx_path)
    if "VIDEO" not in df.columns or "GLOBA_RATING_SCORE" not in df.columns:
        raise ValueError("Excel 文件中缺少 'VIDEO' 或 'GLOBA_RATING_SCORE' 列。")

    # 计算每个视频的平均得分
    video_scores = df.groupby("VIDEO")["GLOBA_RATING_SCORE"].mean().to_dict()
    return video_scores, df


def split_by_student(df, seed=42):
    """根据 STUDENT 划分 70/15/15"""
    students = sorted(df["STUDENT"].unique())
    train_s, temp_s = train_test_split(students, test_size=0.3, random_state=seed, shuffle=True)
    val_s, test_s = train_test_split(temp_s, test_size=0.5, random_state=seed)
    return train_s, val_s, test_s


def get_case_id(case_name):
    """提取 Case_ID 数字部分"""
    import re
    digits = re.findall(r'\d+', case_name)
    return int(digits[0]) if digits else 0


def generate_csv():
    all_entries = []
    global_idx = 0

    # === 读入评分 ===
    video_scores, meta_df = load_video_scores(META_PATH)
    print(f"✅ Loaded {len(video_scores)} videos with average scores")

    # === 训练/验证/测试划分 ===
    train_students, val_students, test_students = split_by_student(meta_df)
    print(f"→ Train Students: {len(train_students)}, Val: {len(val_students)}, Test: {len(test_students)}")

    # 为每个视频找到对应学生和分片类别
    video_to_student = meta_df.groupby("VIDEO")["STUDENT"].first().to_dict()

    # 帧文件夹
    all_videos = sorted(os.listdir(FRAME_ROOT))
    print(f"📂 Found {len(all_videos)} frame folders under {FRAME_ROOT}")

    for video in tqdm(all_videos, desc="Processing videos"):
        frame_dir = os.path.join(FRAME_ROOT, video)
        if not os.path.isdir(frame_dir):
            continue

        # 平均得分
        if video not in video_scores:
            print(f"⚠️ No score found for {video}, skipping.")
            continue
        avg_score = video_scores[video]

        # 判断该视频属于哪个 split
        stu = video_to_student.get(video, None)
        if stu is None:
            print(f"⚠️ No student info for {video}, skipping.")
            continue
        if stu in train_students:
            split = "train"
        elif stu in val_students:
            split = "val"
        elif stu in test_students:
            split = "test"
        else:
            print(f"⚠️ Student {stu} not found in splits for {video}")
            continue

        # 帧文件
        frame_files = sorted(
            [f for f in os.listdir(frame_dir) if f.lower().endswith((".jpg", ".png"))]
        )
        if not frame_files:
            print(f"⚠️ No frames in {frame_dir}, skipping video.")
            continue

        case_id = get_case_id(video)

        for f in frame_files:
            frame_path = os.path.join(frame_dir, f)
            item = {
                "index": global_idx,
                "DataName": DATANAME,
                "Year": YEAR,
                "Case_Name": video,
                "Case_ID": case_id,
                "Frame_Path": frame_path,
                "Phase_GT": avg_score,
                "Phase_Name": PHASE_NAME,
                "Split": split,
            }
            all_entries.append(item)
            global_idx += 1

    df = pd.DataFrame(all_entries)
    print(f"\n✅ Total frames processed: {len(df)}")

    # === 保存 CSV ===
    os.makedirs(OUT_DIR, exist_ok=True)
    for split_name in ["train", "val", "test"]:
        df_split = df[df["Split"] == split_name]
        if not df_split.empty:
            out_csv = os.path.join(OUT_DIR, f"{split_name}_metadata.csv")
            df_split.to_csv(out_csv, index=False)
            print(f"💾 Saved {len(df_split)} frames → {out_csv}")
        else:
            print(f"⚠️ No frames saved for {split_name}")


if __name__ == "__main__":
    generate_csv()

