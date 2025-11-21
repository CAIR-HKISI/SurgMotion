import os
import re
import subprocess
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# === 路径配置 ===
FRAME_ROOT = "data/Surge_Frames/AIxsuture/frames"
META_PATH = "data/Landscopy/AIxsuture/OSATS.xlsx"
OUT_DIR = "data/Surge_Frames/AIxsuture"

YEAR = 2025
DATANAME = "AIxsuture"
PHASE_NAME = "Suturing"


# === 打分与标签映射 ===
def label_from_score(total_score: float):
    """
    根据总分映射 skill 等级标签:
        novice:      [8, 16)   -> 0
        intermediate:[16, 24)  -> 1
        expert:      [24, 41]  -> 2
    超出范围返回 None。
    """
    if total_score is None:
        return None
    s = round(float(total_score))  # 四舍五入到整数，避免浮点误差
    if 8 <= s < 16:
        return 0
    if 16 <= s < 24:
        return 1
    if 24 <= s <= 41:
        return 2
    return None


LABEL_NAME_MAP = {
    0: "novice",
    1: "intermediate",
    2: "expert",
}


# === 函数定义 ===
def load_scores_and_students(xlsx_path):
    """
    从 OSATS.xlsx 读取信息:
      - 每个视频的平均 GLOBA_RATING_SCORE
      - 每个 STUDENT 的平均 GLOBA_RATING_SCORE 及对应标签
      - meta_df: 原始 DataFrame（含 VIDEO / STUDENT / GLOBA_RATING_SCORE）
    """
    df = pd.read_excel(xlsx_path)
    required_cols = {"VIDEO", "STUDENT", "GLOBA_RATING_SCORE"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Excel 文件中缺少列: {required_cols - set(df.columns)}")

    # 每个视频的平均分（可能有多个评分者）
    video_scores = df.groupby("VIDEO")["GLOBA_RATING_SCORE"].mean().to_dict()

    # 每个学生的平均总分
    student_scores = df.groupby("STUDENT")["GLOBA_RATING_SCORE"].mean().to_dict()
    student_labels = {}
    for stu, score in student_scores.items():
        lab = label_from_score(score)
        if lab is None:
            print(f"⚠️ 学生 {stu} 的平均分 {score:.2f} 超出标签映射范围，将跳过其样本。")
        student_labels[stu] = lab

    return video_scores, student_scores, student_labels, df


def split_by_student(df, seed: int = 42):
    """
    按 STUDENT 划分 70% 训练, 15% 验证, 15% 测试，
    且对每个 skill level（0/1/2）分别做 70/15/15，再合并结果。
    """
    rng = seed
    # 先获取每个 STUDENT 的 skill label
    student_labels = (
        df.groupby("STUDENT")["GLOBA_RATING_SCORE"]
        .mean()
        .apply(label_from_score)
        .to_dict()
    )

    level_to_students = {0: [], 1: [], 2: []}
    for stu, lab in student_labels.items():
        if lab is None:
            continue
        if lab in level_to_students:
            level_to_students[lab].append(stu)

    train_s, val_s, test_s = [], [], []

    for level, students in level_to_students.items():
        if not students:
            continue
        students = sorted(students)
        # 对该 level 的学生按 70/30 划分，再对 30 做 1:1 分成 val/test
        lv_train, lv_temp = train_test_split(
            students, test_size=0.3, random_state=rng, shuffle=True
        )
        lv_val, lv_test = train_test_split(
            lv_temp, test_size=0.5, random_state=rng, shuffle=True
        )
        train_s.extend(lv_train)
        val_s.extend(lv_val)
        test_s.extend(lv_test)

    return sorted(train_s), sorted(val_s), sorted(test_s)


def get_case_id(case_name: str) -> int:
    """提取 Case_ID 数字部分（若无数字则为 0）"""
    digits = re.findall(r"\d+", case_name)
    return int(digits[0]) if digits else 0


def videos_to_frames(
    input_path: str = "data/Landscopy/AIxsuture",
    output_path: str = FRAME_ROOT,
    fps: int = 10,
    pattern: str = "*.mp4",
    debug: bool = False,
    save_failed: bool = True,
):
    """
    将 input_path 下的所有 mp4 抽帧到 output_path 下:
      input_path/VIDEO.mp4 -> output_path/VIDEO/VIDEO_%05d.jpg
    """
    input_root = Path(input_path)
    output_root = Path(output_path)
    output_root.mkdir(parents=True, exist_ok=True)

    video_files = sorted(input_root.glob(pattern))

    if not video_files:
        print(f"⚠️ 未在 {input_root} 下找到匹配 {pattern} 的视频文件。")
        return

    print(f"\n🎞️ 共检测 {len(video_files)} 个视频，开始按 {fps} fps 抽帧...\n")

    failed_videos = []

    for vid_path in tqdm(video_files, desc="Extracting frames"):
        vid_stem = vid_path.stem
        out_folder = output_root / vid_stem
        out_folder.mkdir(parents=True, exist_ok=True)
        output_pattern = out_folder / f"{vid_stem}_%05d.jpg"

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(vid_path.resolve()),
            "-safe",
            "0",
            "-vf",
            f"fps={fps},scale=512:-1:flags=bicubic",
            "-vsync",
            "2",
            "-qscale:v",
            "2",
            str(output_pattern),
        ]

        if debug:
            print("🔍 FFmpeg 命令:", " ".join(ffmpeg_cmd))

        try:
            result = subprocess.run(
                ffmpeg_cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if debug:
                print(result.stderr.decode("utf-8", errors="ignore")[:200])
        except subprocess.CalledProcessError as e:
            log = e.stderr.decode("utf-8", errors="ignore")
            print(f"\n❌ 抽帧失败: {vid_path}")
            if debug:
                print("详细错误:\n", log[:400])
            failed_videos.append(str(vid_path))
            continue

    print("\n🎉 抽帧任务完成。")

    if save_failed and failed_videos:
        fail_log = output_root / "failed_videos.txt"
        with fail_log.open("w", encoding="utf-8") as f:
            f.write("\n".join(failed_videos))
        print(f"⚠️ 共 {len(failed_videos)} 个视频抽帧失败，详情见: {fail_log}")


def generate_clip_txt(video_frames_dir: str, txt_path: str) -> int:
    """
    为单个视频帧目录生成 clip 级 txt，写入该视频所有帧的路径。
    返回该视频包含的帧数。
    """
    frame_files = sorted(
        [
            os.path.join(video_frames_dir, f)
            for f in os.listdir(video_frames_dir)
            if os.path.isfile(os.path.join(video_frames_dir, f))
               and f.lower().endswith((".jpg", ".png"))
        ],
        key=lambda p: os.path.basename(p),
    )

    with open(txt_path, "w", encoding="utf-8") as f:
        for frame_path in frame_files:
            f.write(str(frame_path).replace("\\", "/") + "\n")

    return len(frame_files)


def generate_csv():
    all_entries = []
    global_idx = 0

    # === 读入评分 & 学生标签 ===
    (
        video_scores,
        student_scores,
        student_labels,
        meta_df,
    ) = load_scores_and_students(META_PATH)
    print(f"✅ Loaded {len(video_scores)} videos with average scores")
    print(f"✅ Loaded {len(student_scores)} students with average scores & labels")

    # === 训练/验证/测试划分（按 STUDENT） ===
    train_students, val_students, test_students = split_by_student(meta_df)
    print(
        f"→ Train Students: {len(train_students)}, "
        f"Val: {len(val_students)}, Test: {len(test_students)}"
    )

    # 为每个视频找到对应学生
    video_to_student = meta_df.groupby("VIDEO")["STUDENT"].first().to_dict()

    # 帧文件夹
    all_videos = sorted(os.listdir(FRAME_ROOT))
    print(f"📂 Found {len(all_videos)} frame folders under {FRAME_ROOT}")

    for video in tqdm(all_videos, desc="Processing videos"):
        frame_dir = os.path.join(FRAME_ROOT, video)
        if not os.path.isdir(frame_dir):
            continue

        # 视频平均得分
        if video not in video_scores:
            print(f"⚠️ No score found for {video}, skipping.")
            continue
        video_score = float(video_scores[video])

        # 学生信息
        stu = video_to_student.get(video, None)
        if stu is None:
            print(f"⚠️ No student info for {video}, skipping.")
            continue

        student_avg_score = float(student_scores.get(stu, video_score))
        skill_label = student_labels.get(stu, None)
        if skill_label is None:
            print(f"⚠️ Student {stu} has no valid skill label, skipping video {video}.")
            continue

        # 判断该视频属于哪个 split（按 STUDENT）
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
                # === 关键字段：原始得分 & 学生 ID & 学生平均分 & skill 标签 ===
                "Student_ID": stu,
                "Video_Score": video_score,
                "Student_Avg_Score": student_avg_score,
                "Skill_Label": int(skill_label),
                "Skill_Label_Name": LABEL_NAME_MAP.get(skill_label, "unknown"),
                # 保留原有字段以兼容：Phase_GT = 视频平均得分
                "Phase_GT": video_score,
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


def generate_clip_level_csv(
    frame_root: str = FRAME_ROOT,
    meta_path: str = META_PATH,
    out_dir: str = OUT_DIR,
):
    """
    基于已抽好的帧和 OSATS.xlsx 生成【视频 clip 级别】的 CSV，
    同时为每个视频生成一个 txt，包含该视频所有帧的路径。

    输出：
      - data/Surge_Frames/AIxsuture/clip_infos/<VIDEO>.txt
      - data/Surge_Frames/AIxsuture/train_clip_metadata.csv
      - data/Surge_Frames/AIxsuture/val_clip_metadata.csv
      - data/Surge_Frames/AIxsuture/test_clip_metadata.csv
    """
    clip_info_dir = os.path.join(out_dir, "clip_infos")
    os.makedirs(clip_info_dir, exist_ok=True)

    # 读入评分 & 学生标签
    (
        video_scores,
        student_scores,
        student_labels,
        meta_df,
    ) = load_scores_and_students(meta_path)
    print(f"✅ [clips] Loaded {len(video_scores)} videos with average scores")
    print(f"✅ [clips] Loaded {len(student_scores)} students with average scores & labels")

    # 训练/验证/测试划分（按 STUDENT）
    train_students, val_students, test_students = split_by_student(meta_df)
    print(
        f"[clips] → Train Students: {len(train_students)}, "
        f"Val: {len(val_students)}, Test: {len(test_students)}"
    )

    # 为每个视频找到对应学生
    video_to_student = meta_df.groupby("VIDEO")["STUDENT"].first().to_dict()

    all_videos = sorted(os.listdir(frame_root))
    print(f"[clips] 📂 Found {len(all_videos)} frame folders under {frame_root}")

    all_entries = []
    index = 0
    case_id_counter = 0  # 纯数字 case_id

    for video in tqdm(all_videos, desc="Processing videos for clip-level CSV"):
        frame_dir = os.path.join(frame_root, video)
        if not os.path.isdir(frame_dir):
            continue

        # 视频平均得分
        if video not in video_scores:
            print(f"[clips] ⚠️ No score found for {video}, skipping.")
            continue
        video_score = float(video_scores[video])

        # 学生信息
        stu = video_to_student.get(video, None)
        if stu is None:
            print(f"[clips] ⚠️ No student info for {video}, skipping.")
            continue

        student_avg_score = float(student_scores.get(stu, video_score))
        skill_label = student_labels.get(stu, None)
        if skill_label is None:
            print(f"[clips] ⚠️ Student {stu} has no valid skill label, skipping video {video}.")
            continue

        # 判断该视频属于哪个 split（按 STUDENT）
        if stu in train_students:
            split = "train"
        elif stu in val_students:
            split = "val"
        elif stu in test_students:
            split = "test"
        else:
            print(f"[clips] ⚠️ Student {stu} not found in splits for {video}")
            continue

        # 生成该视频的 txt（一个 clip，clip_idx=0）
        txt_path = os.path.join(clip_info_dir, f"{video}.txt")
        num_frames = generate_clip_txt(frame_dir, txt_path)
        if num_frames == 0:
            print(f"[clips] ⚠️ No frames in {frame_dir}, skipping video.")
            continue

        case_id = case_id_counter  # 全局递增整数
        clip_idx = 0

        item = {
            "Index": index,
            "clip_path": str(txt_path).replace("\\", "/"),
            "label": int(skill_label),
            "label_name": LABEL_NAME_MAP.get(int(skill_label), "unknown"),
            "case_id": case_id,
            "clip_idx": clip_idx,
            # 额外信息：方便分析
            "Student_ID": stu,
            "Video_Score": video_score,
            "Student_Avg_Score": student_avg_score,
            "Split": split,
        }
        all_entries.append(item)
        index += 1
        case_id_counter += 1

    df = pd.DataFrame(all_entries)
    print(f"\n[clips] ✅ Total clips (videos) processed: {len(df)}")

    # 按 Split 保存三个 CSV
    for split_name in ["train", "val", "test"]:
        df_split = df[df["Split"] == split_name]
        if not df_split.empty:
            out_csv = os.path.join(out_dir, f"{split_name}_clip_metadata.csv")
            df_split.to_csv(out_csv, index=False)
            print(f"[clips] 💾 Saved {len(df_split)} clips → {out_csv}")
        else:
            print(f"[clips] ⚠️ No clips saved for {split_name}")


def main():
    parser = argparse.ArgumentParser(
        description="AIxsuture 预处理：可选抽帧 + 帧级 CSV + 视频 clip 级 txt/CSV"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/Landscopy/AIxsuture",
        help="原始 AIxsuture 视频目录（用于抽帧，可选）",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=FRAME_ROOT,
        help="抽帧输出目录 / 已存在的帧目录",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=1,
        help="抽帧帧率（不需要重抽时可忽略）",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="是否打印 ffmpeg 调试信息",
    )
    parser.add_argument(
        "--no_extract",
        action="store_true",
        help="若指定，则不执行抽帧，只基于已有帧生成 CSV/txt",
    )
    args = parser.parse_args()

    # Step 1: （可选）抽帧
    if not args.no_extract:
        print("\n🎞️ 开始为 AIxsuture 抽帧 ...")
        videos_to_frames(
            input_path=args.input_path,
            output_path=args.output_path,
            fps=args.fps,
            debug=args.debug,
        )
    else:
        print("\n⚠️ 跳过抽帧步骤，仅使用已有帧目录生成 CSV/txt")

    # Step 2: 生成帧级 CSV
    print("\n📄 开始基于 OSATS 打分生成 AIxsuture 帧级元数据 CSV ...")
    generate_csv()

    # Step 3: 生成视频 clip 级 txt + CSV
    print("\n📄 开始生成 AIxsuture 视频 clip 级 txt 与 CSV ...")
    generate_clip_level_csv()


if __name__ == "__main__":
    import argparse

    main()


