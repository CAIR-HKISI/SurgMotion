#!/usr/bin/env python3
"""
重新生成SurgAction160的CSV文件
对每个独立的视频片段采样64帧，每个视频片段一行
"""

import os
import csv
import re
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# ========================================
# 配置路径
# ========================================
FRAMES_BASE = "data/Surge_Frames/SurgAction160/frames"
OUT_DIR = "data/Surge_Frames/SurgAction160/clips_64f"
TXT_OUT_DIR = os.path.join(OUT_DIR, "clip_dense_64f_info_v2")
TARGET_FRAMES = 64

# 从现有CSV文件中读取train/val/test划分信息
OLD_TRAIN_CSV = "data/Surge_Frames/SurgAction160/clips_64f/train_dense_64f_detailed.csv"
OLD_VAL_CSV = "data/Surge_Frames/SurgAction160/clips_64f/val_dense_64f_detailed.csv"
OLD_TEST_CSV = "data/Surge_Frames/SurgAction160/clips_64f/test_dense_64f_detailed.csv"

# 新的CSV文件路径
NEW_TRAIN_CSV = "data/Surge_Frames/SurgAction160/clips_64f/train_dense_64f_detailed.csv"
NEW_VAL_CSV = "data/Surge_Frames/SurgAction160/clips_64f/val_dense_64f_detailed.csv"
NEW_TEST_CSV = "data/Surge_Frames/SurgAction160/clips_64f/test_dense_64f_detailed.csv"


def extract_case_id_from_path(video_dir_name):
    """从视频目录名称（如 '01_01'）提取数字ID"""
    numbers = re.findall(r'\d+', video_dir_name)
    if numbers:
        return int(''.join(numbers))
    return 0


def get_action_name_from_folder(folder_name):
    """从文件夹名称（如 '01_abdominal_access'）提取动作名称"""
    parts = folder_name.split('_', 1)
    if len(parts) > 1:
        return parts[1]
    return folder_name


def sample_frames(frame_files, target_num=TARGET_FRAMES):
    """
    对帧文件列表进行采样
    
    Args:
        frame_files: 排序后的帧文件路径列表
        target_num: 目标帧数（默认64）
    
    Returns:
        采样后的帧文件路径列表
    """
    total_frames = len(frame_files)
    
    if total_frames == 0:
        return []
    
    if total_frames <= target_num:
        # 不足64帧，重复最后一帧
        sampled = frame_files.copy()
        last_frame = frame_files[-1]
        while len(sampled) < target_num:
            sampled.append(last_frame)
        return sampled
    else:
        # 超过64帧，间隔采样
        indices = []
        step = total_frames / target_num
        for i in range(target_num):
            idx = int(i * step)
            if idx >= total_frames:
                idx = total_frames - 1
            indices.append(idx)
        return [frame_files[i] for i in indices]


def load_split_info():
    """从现有CSV文件中加载train/val/test划分信息"""
    split_info = {}  # (case_id, clip_idx) -> split
    
    for split_name, csv_path in [('train', OLD_TRAIN_CSV), ('val', OLD_VAL_CSV), ('test', OLD_TEST_CSV)]:
        if not os.path.exists(csv_path):
            print(f"警告: {csv_path} 不存在，跳过")
            continue
            
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                case_id = int(row['case_id'])
                clip_idx = int(row['clip_idx'])
                key = (case_id, clip_idx)
                split_info[key] = split_name
    
    return split_info


def load_label_mapping():
    """从现有CSV文件中加载标签映射"""
    label_map = {}  # label_name -> label
    
    for csv_path in [OLD_TRAIN_CSV, OLD_VAL_CSV, OLD_TEST_CSV]:
        if not os.path.exists(csv_path):
            continue
            
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                label_name = row['label_name']
                label = int(row['label'])
                if label_name not in label_map:
                    label_map[label_name] = label
    
    return label_map


def generate_csv():
    """生成新的CSV文件"""
    # 创建输出目录
    os.makedirs(TXT_OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(NEW_TRAIN_CSV), exist_ok=True)
    
    # 加载划分信息和标签映射
    print("加载train/val/test划分信息...")
    split_info = load_split_info()
    print(f"加载了 {len(split_info)} 个视频片段的划分信息")
    
    print("加载标签映射...")
    label_map = load_label_mapping()
    print(f"标签映射: {label_map}")
    
    # 存储每个split的数据
    split_data = {'train': [], 'val': [], 'test': []}
    
    # 遍历所有手术动作文件夹
    frames_base = Path(FRAMES_BASE)
    if not frames_base.exists():
        print(f"错误: {FRAMES_BASE} 不存在")
        return
    
    action_folders = sorted([d for d in frames_base.iterdir() if d.is_dir()])
    print(f"找到 {len(action_folders)} 个手术动作文件夹")
    
    global_idx = 0
    
    for action_folder in tqdm(action_folders, desc="处理手术动作"):
        action_name = get_action_name_from_folder(action_folder.name)
        label = label_map.get(action_name, -1)
        
        if label == -1:
            print(f"警告: 动作 {action_name} 没有找到标签映射")
            continue
        
        # 遍历该动作下的所有视频片段
        video_clips = sorted([d for d in action_folder.iterdir() if d.is_dir()])
        
        for video_clip in video_clips:
            # 获取所有帧文件
            frame_files = sorted([f for f in video_clip.iterdir() 
                                 if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
            
            if len(frame_files) == 0:
                continue
            
            # 采样64帧
            sampled_frames = sample_frames(frame_files, TARGET_FRAMES)
            
            # 提取case_id和clip_idx
            # 视频片段名称格式：如 "01_01", "01_02" 等
            clip_name = video_clip.name
            case_id = extract_case_id_from_path(clip_name)
            # 从clip_name中提取clip_idx（第二个数字）
            clip_idx_match = re.search(r'_(\d+)$', clip_name)
            if clip_idx_match:
                clip_idx = int(clip_idx_match.group(1)) - 1  # 转换为0-based索引
            else:
                clip_idx = 0
            
            # 确定split
            split = split_info.get((case_id, clip_idx), 'train')  # 默认train
            
            # 生成txt文件名
            txt_filename = f"{video_clip.name}_64f.txt"
            txt_path = os.path.join(TXT_OUT_DIR, txt_filename)
            txt_relative_path = f"data/Surge_Frames/SurgAction160/clips_64f/clip_dense_64f_info_v2/{txt_filename}"
            
            # 保存txt文件
            with open(txt_path, 'w', encoding='utf-8') as f:
                for frame_path in sampled_frames:
                    # 转换为相对路径（从项目根目录开始）
                    relative_frame_path = str(frame_path).replace(os.path.abspath('.'), '').lstrip('/')
                    if not relative_frame_path.startswith('data/'):
                        # 如果路径不对，尝试直接使用相对路径
                        relative_frame_path = str(frame_path.relative_to(Path('.').resolve()))
                    f.write(relative_frame_path + '\n')
            
            # 计算帧数信息
            actual_frames = len(frame_files)
            padded_frames = TARGET_FRAMES - actual_frames if actual_frames < TARGET_FRAMES else 0
            is_padded = actual_frames < TARGET_FRAMES
            
            # 计算时间信息（假设30fps）
            fps = 30.0
            duration_seconds = TARGET_FRAMES / fps
            start_time = "0:00:00"
            end_time_seconds = duration_seconds
            hours = int(end_time_seconds // 3600)
            minutes = int((end_time_seconds % 3600) // 60)
            seconds = int(end_time_seconds % 60)
            end_time = f"{hours}:{minutes:02d}:{seconds:02d}"
            
            # 创建CSV行数据
            row_data = {
                'Index': global_idx,
                'clip_path': txt_relative_path,
                'label': label,
                'label_name': action_name,
                'case_id': case_id,
                'clip_idx': clip_idx,
                'start_frame': 0,
                'end_frame': actual_frames - 1,
                'actual_frames': actual_frames,
                'padded_frames': padded_frames,
                'start_time': start_time,
                'end_time': end_time,
                'duration_seconds': duration_seconds,
                'is_padded': is_padded
            }
            
            split_data[split].append(row_data)
            global_idx += 1
    
    # 保存CSV文件
    csv_columns = ['Index', 'clip_path', 'label', 'label_name', 'case_id', 'clip_idx',
                   'start_frame', 'end_frame', 'actual_frames', 'padded_frames',
                   'start_time', 'end_time', 'duration_seconds', 'is_padded']
    
    for split_name in ['train', 'val', 'test']:
        csv_path = [NEW_TRAIN_CSV, NEW_VAL_CSV, NEW_TEST_CSV][['train', 'val', 'test'].index(split_name)]
        
        # 按Index排序
        split_data[split_name].sort(key=lambda x: x['Index'])
        
        # 重新分配Index
        for idx, row in enumerate(split_data[split_name]):
            row['Index'] = idx
        
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writeheader()
            writer.writerows(split_data[split_name])
        
        print(f"保存 {split_name} CSV: {csv_path}, 共 {len(split_data[split_name])} 行")
    
    print("\n完成！")
    print(f"训练集: {len(split_data['train'])} 个视频片段")
    print(f"验证集: {len(split_data['val'])} 个视频片段")
    print(f"测试集: {len(split_data['test'])} 个视频片段")
    print(f"TXT文件保存在: {TXT_OUT_DIR}")


if __name__ == "__main__":
    generate_csv()

