import os
import pandas as pd
from datetime import timedelta
from tqdm import tqdm


def process_unlabeled_csv_dense_sampling(
    input_csv_path,
    output_csv_path,
    clip_info_dir,
    window_size=16,
    stride=1,
    fps=1,
):
    """
    从 unlabeled_metadata.csv 生成 dense clips。
    - 不区分 train/val/test，仅处理 unlabeled 数据集
    - 若帧数不足窗口窗口大小，则重复最后一帧补齐
    - 输出每个 clip 的帧路径 txt 文件 + detailed CSV
    """

    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"❌ 找不到输入文件: {input_csv_path}")

    df = pd.read_csv(input_csv_path)
    if "Case_ID" not in df.columns or "Frame_Path" not in df.columns:
        raise ValueError("❌ 输入 CSV 必须包含列 'Case_ID' 和 'Frame_Path'")

    os.makedirs(clip_info_dir, exist_ok=True)

    frames_per_window = window_size * fps
    frames_per_stride = stride * fps
    all_clips_data = []

    # 按视频分组
    video_groups = df.groupby("Case_ID")

    print(f"📄 输入文件: {input_csv_path}")
    print(f"⚙️ 参数: window={window_size}s, stride={stride}s, fps={fps}")
    print("=================================")

    # 遍历每个视频
    for case_id, video_df in tqdm(video_groups, desc="Processing videos"):
        video_df = video_df.sort_values("Frame_Path").reset_index(drop=True)
        total_frames = len(video_df)
        clip_count = 0
        start_idx = 0

        # 滑动窗口生成 clip
        while start_idx < total_frames:
            end_idx = min(start_idx + frames_per_window, total_frames)
            clip_frames = video_df.iloc[start_idx:end_idx].copy()
            actual_frames = len(clip_frames)

            # 填充不足帧
            is_padded = actual_frames < frames_per_window
            padded_frames = 0
            if is_padded:
                last_row = clip_frames.iloc[-1]
                padded_frames = frames_per_window - actual_frames
                padding_df = pd.DataFrame([last_row.to_dict()] * padded_frames)
                clip_frames = pd.concat([clip_frames, padding_df], ignore_index=True)

            # clip 文件名
            clip_identifier = f"case{case_id}_c{clip_count:03d}_f{start_idx:06d}-{end_idx:06d}"
            if is_padded:
                clip_identifier += "_padded"

            clip_txt_path = os.path.join(clip_info_dir, f"{clip_identifier}.txt")

            # 写入 txt（记录帧路径）
            with open(clip_txt_path, "w") as f:
                for _, row in clip_frames.iterrows():
                    f.write(f"{row['Frame_Path']}\n")

            # 记录 clip 信息
            clip_info = {
                "clip_path": clip_txt_path,
                "case_id": case_id,
                "clip_idx": clip_count,
                "start_frame": start_idx,
                "end_frame": end_idx,
                "actual_frames": actual_frames,
                "padded_frames": padded_frames,
                "is_padded": is_padded,
                "duration_seconds": frames_per_window / fps,
                "start_time": str(timedelta(seconds=int(start_idx / fps))),
                "end_time": str(timedelta(seconds=int(end_idx / fps))),
            }
            all_clips_data.append(clip_info)

            start_idx += frames_per_stride
            clip_count += 1

        print(f"📹 Case {case_id}: {clip_count} 个 clip，{total_frames} 帧")

    # 保存详细 CSV
    output_df = pd.DataFrame(all_clips_data)
    detailed_path = output_csv_path.replace(".csv", "_detailed.csv")
    output_df.to_csv(detailed_path, index=False)

    print("=================================")
    print(f"✅ 生成完成，共 {len(output_df)} 个 clip")
    print(f"📁 输出 CSV: {detailed_path}")
    print(f"📂 clip txt 存放目录: {clip_info_dir}")
    print("=================================")
    return output_df


def process_unlabeled_dataset(base_data_path, window_size=16, stride=1):
    """
    针对 unlabeled 数据集生成 dense clip，结构与 train/val 脚本一致
    """
    input_csv = os.path.join(base_data_path, "unlabeled_metadata.csv")
    output_base_path = os.path.join(base_data_path, f"clips_{window_size}f")
    output_csv = os.path.join(output_base_path, f"unlabeled_dense_{window_size}f.csv")
    clip_info_dir = os.path.join(output_base_path, f"clip_dense_{window_size}f_info")

    os.makedirs(output_base_path, exist_ok=True)

    df = process_unlabeled_csv_dense_sampling(
        input_csv_path=input_csv,
        output_csv_path=output_csv,
        clip_info_dir=clip_info_dir,
        window_size=window_size,
        stride=stride,
        fps=1,
    )

    if "is_padded" in df.columns:
        padded = df["is_padded"].sum()
        print(f"\n📊 填充 clip 数量: {padded}/{len(df)} ({padded/len(df)*100:.1f}%)")

    print(f"📊 总 clip 数量: {len(df)}")
    print(f"✅ 数据集 {os.path.basename(base_data_path)} - window={window_size}f, stride={stride}s 处理完成。")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate clips from unlabeled dataset with customizable stride."
    )
    parser.add_argument(
        "--base_data_path",
        type=str,
        required=True,
        help="数据集根目录，例如 data/Surge_Frames/AutoLaparo",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=64,
        help="窗口大小 (默认=64)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="滑动步长 (默认=1)",
    )

    args = parser.parse_args()
    base_data_path = args.base_data_path
    window_size = args.window_size
    stride = args.stride

    print(f"\n###### 处理数据集 {base_data_path}，窗口大小: {window_size}f，步长: {stride}s ######")
    process_unlabeled_dataset(base_data_path, window_size, stride)

