import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from datetime import timedelta


def extract_frames_from_video(video_path, output_dir, fps=15):
    """从视频提取帧（15fps）"""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_name = video_path.stem  # 不带扩展名的文件名
    out_folder = output_dir / file_name
    out_folder.mkdir(parents=True, exist_ok=True)
    
    # 输出文件名前缀
    output_pattern = out_folder / f"{file_name}_%08d.jpg"
    
    # 调用 ffmpeg 提取帧
    ffmpeg_cmd = (
        f'ffmpeg -y -i "{video_path}" '
        f'-vf "fps={fps}" "{output_pattern}"'
    )
    os.system(ffmpeg_cmd)
    
    return out_folder


def sample_frames_to_64(frame_dir, output_txt_path):
    """
    对视频帧采样到64帧
    - 如果不足64帧，重复最后一帧到64帧
    - 如果超过64帧，间隔采样64帧
    """
    frame_dir = Path(frame_dir)
    output_txt_path = Path(output_txt_path)
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 获取所有帧文件，按文件名排序
    frame_files = sorted(frame_dir.glob("*.jpg"))
    
    if len(frame_files) == 0:
        print(f"警告: {frame_dir} 中没有找到帧文件")
        return False, 0
    
    # 采样逻辑
    if len(frame_files) <= 64:
        # 不足64帧，重复最后一帧
        selected_frames = list(frame_files)
        last_frame = frame_files[-1]
        padding_count = 64 - len(frame_files)
        selected_frames.extend([last_frame] * padding_count)
        actual_frames = len(frame_files)
        padded_frames = padding_count
    else:
        # 超过64帧，间隔采样
        # 使用等间隔采样，确保均匀分布
        indices = np.linspace(0, len(frame_files) - 1, 64, dtype=int)
        selected_frames = [frame_files[i] for i in indices]
        actual_frames = len(frame_files)
        padded_frames = 0
    
    # 保存路径到txt文件
    with open(output_txt_path, 'w') as f:
        for frame_path in selected_frames:
            f.write(f"{frame_path}\n")
    
    return True, actual_frames, padded_frames


def load_split_labels(split_file):
    """加载split文件中的标签信息"""
    labels = {}
    with open(split_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 2:
                print(f"警告: 第{line_num}行格式不正确，跳过: {line}")
                continue
            
            # 尝试解析标签，处理可能的格式问题
            try:
                # 通常格式是: 视频名,标签
                video_name_with_ext = parts[0].strip()
                label_str = parts[1].strip()
                
                # 如果标签是字符串，尝试转换
                if label_str.isdigit():
                    label = int(label_str)
                elif label_str.lower() == 'abnormal' or label_str == '1':
                    label = 1
                elif label_str.lower() == 'normal' or label_str == '0':
                    label = 0
                else:
                    print(f"警告: 第{line_num}行标签格式不正确: {label_str}，跳过")
                    continue
                
                # 移除扩展名，用于匹配
                video_name = Path(video_name_with_ext).stem
                # 同时保存带扩展名和不带扩展名的版本
                labels[video_name_with_ext] = label
                labels[video_name] = label
            except Exception as e:
                print(f"错误: 第{line_num}行解析失败: {line}")
                print(f"  错误信息: {e}")
                continue
    return labels


def process_polypdiag_dataset(
    video_dir,
    frame_output_dir,
    clip_txt_output_dir,
    splits_dir,
    target_frames=64,
    extract_fps=15,
    skip_extraction=False
):
    """
    处理PolypDiag数据集的完整流程：
    1. 提取视频帧（15fps）- 如果skip_extraction=True则跳过
    2. 采样到64帧并保存路径到txt
    3. 生成CSV文件
    """
    video_dir = Path(video_dir)
    frame_output_dir = Path(frame_output_dir)
    clip_txt_output_dir = Path(clip_txt_output_dir)
    splits_dir = Path(splits_dir)
    
    # 创建输出目录
    frame_output_dir.mkdir(parents=True, exist_ok=True)
    clip_txt_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载train和val的标签
    print("正在加载标签文件...")
    train_labels = load_split_labels(splits_dir / "train.txt")
    print(f"加载训练集标签: {len(train_labels)} 个")
    val_labels = load_split_labels(splits_dir / "val.txt")
    print(f"加载验证集标签: {len(val_labels)} 个")
    
    # 合并所有标签
    all_labels = {**train_labels, **val_labels}
    print(f"总共加载 {len(all_labels)} 个标签")
    
    # 获取所有视频文件
    video_files = list(video_dir.glob("*.mp4"))
    print(f"找到 {len(video_files)} 个视频文件")
    
    # 创建视频名到数字的映射（case_id）
    video_name_to_case_id = {}
    case_id_counter = 0
    for video_path in video_files:
        video_name = video_path.stem
        video_name_with_ext = video_path.name
        # 检查是否在标签文件中
        if video_name_with_ext in all_labels or video_name in all_labels:
            if video_name not in video_name_to_case_id:
                video_name_to_case_id[video_name] = case_id_counter
                case_id_counter += 1
    
    print(f"创建了 {len(video_name_to_case_id)} 个视频的case_id映射")
    
    all_clips_data = []
    
    # 处理每个视频
    for video_path in tqdm(video_files, desc="处理视频"):
        video_name = video_path.stem
        video_name_with_ext = video_path.name
        
        # 获取标签（如果存在）- 优先使用带扩展名的版本，然后是不带扩展名的版本
        label = None
        if video_name_with_ext in all_labels:
            label = all_labels[video_name_with_ext]
        elif video_name in all_labels:
            label = all_labels[video_name]
        else:
            print(f"警告: 视频 {video_name} 在splits文件中没有找到标签，跳过")
            continue
        
        # 获取case_id（数字映射）
        case_id = video_name_to_case_id.get(video_name)
        if case_id is None:
            print(f"警告: 视频 {video_name} 没有对应的case_id，跳过")
            continue
        
        label_name = "abnormal" if label == 1 else "normal"
        
        # 步骤1: 提取视频帧（15fps）- 如果skip_extraction=True则跳过，或帧目录已存在则跳过
        frame_dir = frame_output_dir / video_name
        
        if not skip_extraction:
            # 检查帧目录是否已存在
            if frame_dir.exists() and any(frame_dir.glob("*.jpg")):
                print(f"跳过: 视频 {video_name} 的帧已存在，跳过提取")
            else:
                print(f"提取视频帧: {video_name}")
                frame_dir = extract_frames_from_video(
                    video_path, 
                    frame_output_dir, 
                    fps=extract_fps
                )
        else:
            # 直接使用已存在的帧目录
            if not frame_dir.exists():
                print(f"警告: 视频 {video_name} 的帧目录不存在: {frame_dir}，跳过")
                continue
        
        # 步骤2: 采样到64帧并保存路径到txt
        clip_txt_file = clip_txt_output_dir / f"{video_name}_64f.txt"
        success, actual_frames, padded_frames = sample_frames_to_64(frame_dir, clip_txt_file)
        
        if not success:
            print(f"警告: 视频 {video_name} 采样失败，跳过")
            continue
        
        # 步骤3: 生成clip信息（参考gen_clips.py的结构）
        is_padded = padded_frames > 0  # 如果有padding，说明不足64帧
        
        clip_info = {
            'clip_path': str(clip_txt_file),
            'label': label,
            'label_name': label_name,
            'case_id': case_id,  # 使用数字映射作为case_id
            'case_name': video_name,  # 保存原始视频名
            'clip_idx': 0,  # 每个视频只有一个clip
            'start_frame': 0,
            'end_frame': actual_frames,
            'actual_frames': actual_frames,
            'padded_frames': padded_frames,
            'start_time': '0:00:00',
            'end_time': str(timedelta(seconds=int(actual_frames / extract_fps))),
            'duration_seconds': actual_frames / extract_fps,
            'is_padded': is_padded
        }
        all_clips_data.append(clip_info)
    
    # 保存CSV文件
    output_df = pd.DataFrame(all_clips_data)
    
    # 分别保存train和val的CSV
    train_clips = []
    val_clips = []
    
    for clip in all_clips_data:
        case_name = clip['case_name']  # 使用视频名来匹配
        case_name_with_ext = f"{case_name}.mp4"
        is_train = (case_name in train_labels) or (case_name_with_ext in train_labels)
        is_val = (case_name in val_labels) or (case_name_with_ext in val_labels)
        
        if is_train:
            train_clips.append(clip)
        elif is_val:
            val_clips.append(clip)
    
    train_df = pd.DataFrame(train_clips)
    val_df = pd.DataFrame(val_clips)
    
    # 保存CSV文件 - 放在clip_txt_output_dir的父目录下
    csv_output_dir = clip_txt_output_dir.parent
    csv_output_dir.mkdir(parents=True, exist_ok=True)
    
    train_csv_path = csv_output_dir / "train_dense_64f_detailed.csv"
    val_csv_path = csv_output_dir / "val_dense_64f_detailed.csv"
    all_csv_path = csv_output_dir / "all_dense_64f_detailed.csv"
    
    # 确保case_id和label是整数类型
    train_df['case_id'] = train_df['case_id'].astype(int)
    train_df['label'] = train_df['label'].astype(int)
    val_df['case_id'] = val_df['case_id'].astype(int)
    val_df['label'] = val_df['label'].astype(int)
    output_df['case_id'] = output_df['case_id'].astype(int)
    output_df['label'] = output_df['label'].astype(int)
    
    train_df.to_csv(train_csv_path, index=True, index_label="Index")
    val_df.to_csv(val_csv_path, index=True, index_label="Index")
    output_df.to_csv(all_csv_path, index=True, index_label="Index")
    
    print(f"\n=== 处理完成 ===")
    print(f"总共处理 {len(all_clips_data)} 个视频片段")
    print(f"训练集: {len(train_clips)} 个片段")
    print(f"验证集: {len(val_clips)} 个片段")
    print(f"输出文件:")
    print(f"  - 训练集CSV: {train_csv_path}")
    print(f"  - 验证集CSV: {val_csv_path}")
    print(f"  - 全部CSV: {all_csv_path}")
    
    # 打印统计信息
    print(f"\n标签统计:")
    print(f"训练集 - abnormal: {train_df[train_df['label'] == 1].shape[0]}, normal: {train_df[train_df['label'] == 0].shape[0]}")
    print(f"验证集 - abnormal: {val_df[val_df['label'] == 1].shape[0]}, normal: {val_df[val_df['label'] == 0].shape[0]}")
    
    return output_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="处理PolypDiag数据集：提取帧、采样到64帧、生成CSV"
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="data/GI_Videos/PolypDiag/videos",
        help="视频文件目录"
    )
    parser.add_argument(
        "--frame_output_dir",
        type=str,
        default="data/Surge_Frames/PolypDiag/frames",
        help="视频帧输出目录"
    )
    parser.add_argument(
        "--clip_txt_output_dir",
        type=str,
        default="data/Surge_Frames/PolypDiag/clips_64f/clip_dense_64f_info",
        help="64帧采样后的txt文件输出目录"
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default="data/GI_Videos/PolypDiag/splits",
        help="训练/验证划分文件目录"
    )
    parser.add_argument(
        "--target_frames",
        type=int,
        default=64,
        help="目标帧数（默认64）"
    )
    parser.add_argument(
        "--extract_fps",
        type=int,
        default=15,
        help="提取视频帧的fps（默认15）"
    )
    parser.add_argument(
        "--skip_extraction",
        action="store_true",
        help="跳过视频帧提取步骤（如果帧已经提取完成）"
    )
    
    args = parser.parse_args()
    
    process_polypdiag_dataset(
        video_dir=args.video_dir,
        frame_output_dir=args.frame_output_dir,
        clip_txt_output_dir=args.clip_txt_output_dir,
        splits_dir=args.splits_dir,
        target_frames=args.target_frames,
        extract_fps=args.extract_fps,
        skip_extraction=args.skip_extraction
    )

