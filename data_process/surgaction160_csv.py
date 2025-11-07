import os
import pandas as pd
import re
from pathlib import Path
from tqdm import tqdm
import random

# ========================================
# 基础路径配置
# ========================================
FRAME_PATH = "data/Surge_Frames/SurgAction160/frames"
OUT_DIR = "data/Surge_Frames/SurgAction160"

# 数据名称和年份
DATANAME = "SurgAction160"
YEAR = 2024

# 数据划分比例（用户要求：75%训练，15%验证，20%测试）
# 由于总和为110%，我们按比例归一化到100%
TRAIN_RATIO = 0.75 / 1.10  # ≈ 0.682 (实际约68.2%)
VAL_RATIO = 0.15 / 1.10    # ≈ 0.136 (实际约13.6%)
TEST_RATIO = 0.20 / 1.10   # ≈ 0.182 (实际约18.2%)
# 或者使用更常见的划分：75%训练，15%验证，10%测试
# TRAIN_RATIO = 0.75
# VAL_RATIO = 0.15
# TEST_RATIO = 0.10

# 随机种子（确保每次划分一致）
RANDOM_SEED = 42


def extract_case_id(video_name):
    """
    从视频名称（如 "01_01"）提取数字ID
    """
    # 提取所有数字并组合
    numbers = re.findall(r'\d+', video_name)
    if numbers:
        # 如果有多个数字，组合它们
        return int(''.join(numbers))
    return 0


def get_phase_name_from_folder(folder_name):
    """
    从文件夹名称提取类别名称
    例如: "01_abdominal_access" -> "abdominal_access"
    """
    # 移除前缀数字和下划线
    parts = folder_name.split('_', 1)
    if len(parts) > 1:
        return parts[1]
    return folder_name


def assign_split(video_idx, total_videos, train_ratio, val_ratio, test_ratio):
    """
    根据视频索引和比例分配数据集划分
    
    Args:
        video_idx: 视频索引（从0开始）
        total_videos: 总视频数
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    
    Returns:
        "train", "val", 或 "test"
    """
    train_end = int(total_videos * train_ratio)
    val_end = train_end + int(total_videos * val_ratio)
    
    if video_idx < train_end:
        return "train"
    elif video_idx < val_end:
        return "val"
    else:
        return "test"


def generate_csv():
    """生成CSV文件"""
    frames_base = Path(FRAME_PATH)
    
    if not frames_base.exists():
        print(f"❌ 错误: 帧目录不存在: {frames_base}")
        return
    
    all_data = []
    global_idx = 0
    
    # 获取所有类别文件夹（按名称排序）
    category_folders = sorted([f for f in os.listdir(frames_base) 
                              if os.path.isdir(frames_base / f)])
    
    print(f"找到 {len(category_folders)} 个类别文件夹")
    
    # 创建类别名称到ID的映射
    category_to_id = {}
    phase_id = 0
    
    for category_folder in tqdm(category_folders, desc="处理类别"):
        category_path = frames_base / category_folder
        phase_name = get_phase_name_from_folder(category_folder)
        
        # 为每个类别分配ID
        if phase_name not in category_to_id:
            category_to_id[phase_name] = phase_id
            phase_id += 1
        
        phase_gt = category_to_id[phase_name]
        
        # 获取该类别下的所有视频文件夹
        video_folders = sorted([f for f in os.listdir(category_path)
                                if os.path.isdir(category_path / f)])
        
        if not video_folders:
            print(f"⚠️  警告: 类别 {category_folder} 没有视频文件夹")
            continue
        
        # 设置随机种子以确保每次划分一致
        random.seed(RANDOM_SEED)
        # 随机打乱视频顺序（但使用固定种子确保一致性）
        video_folders_shuffled = video_folders.copy()
        random.shuffle(video_folders_shuffled)
        
        # 为每个视频分配数据集划分
        total_videos = len(video_folders_shuffled)
        
        for video_idx, video_folder in enumerate(video_folders_shuffled):
            video_path = category_path / video_folder
            case_name = video_folder  # 例如 "01_01"
            case_id = extract_case_id(video_folder)
            
            # 分配数据集划分
            split = assign_split(video_idx, total_videos, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
            
            # 获取该视频的所有帧文件
            frame_files = sorted([f for f in os.listdir(video_path)
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
            if not frame_files:
                print(f"⚠️  警告: 视频 {video_folder} 没有帧文件")
                continue
            
            # 为每帧创建数据条目
            for frame_file in frame_files:
                frame_path = video_path / frame_file
                frame_path_str = str(frame_path)
                
                # 检查文件是否存在
                if not frame_path.exists():
                    continue
                
                data_item = {
                    'Index': global_idx,
                    'DataName': DATANAME,
                    'Year': YEAR,
                    'Case_Name': case_name,
                    'Case_ID': case_id,
                    'Frame_Path': frame_path_str,
                    'Phase_GT': phase_gt,
                    'Phase_Name': phase_name,
                    'Split': split
                }
                all_data.append(data_item)
                global_idx += 1
    
    # 转换为DataFrame
    df_all = pd.DataFrame(all_data)
    
    print(f"\n✅ 总共处理了 {len(df_all)} 帧")
    print(f"   - 训练集: {len(df_all[df_all['Split'] == 'train'])} 帧")
    print(f"   - 验证集: {len(df_all[df_all['Split'] == 'val'])} 帧")
    print(f"   - 测试集: {len(df_all[df_all['Split'] == 'test'])} 帧")
    
    # 创建输出目录
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 分别保存train、val和test
    print("\n" + "="*60)
    print("📊 GT 分布统计")
    print("="*60)
    
    for split in ["train", "val", "test"]:
        df_split = df_all[df_all["Split"] == split]
        if not df_split.empty:
            out_csv = os.path.join(OUT_DIR, f"{split}_metadata.csv")
            df_split.to_csv(out_csv, index=False)
            print(f"\n💾 已保存 {len(df_split)} 帧到 {out_csv}")
            
            # 统计该split的GT分布
            print(f"\n📊 {split.upper()} 集 GT 分布:")
            phase_gt_counts = df_split['Phase_GT'].value_counts().sort_index()
            total_frames = len(df_split)
            for phase_gt in sorted(phase_gt_counts.index):
                phase_name = df_split[df_split['Phase_GT'] == phase_gt]['Phase_Name'].iloc[0]
                count = phase_gt_counts[phase_gt]
                percentage = (count / total_frames) * 100
                print(f"   - Phase_GT={phase_gt}: {phase_name}: {count} 帧 ({percentage:.2f}%)")
        else:
            print(f"⚠️  {split} 集没有数据")
    
    # 打印整体统计信息
    print("\n📊 整体动作标签统计:")
    phase_gt_counts = df_all['Phase_GT'].value_counts().sort_index()
    total_frames = len(df_all)
    for phase_gt in sorted(phase_gt_counts.index):
        phase_name = df_all[df_all['Phase_GT'] == phase_gt]['Phase_Name'].iloc[0]
        count = phase_gt_counts[phase_gt]
        percentage = (count / total_frames) * 100
        print(f"   - Phase_GT={phase_gt}: {phase_name}: {count} 帧 ({percentage:.2f}%)")
    
    # 显示类别映射关系
    print("\n🔄 类别映射关系:")
    for phase_name, phase_id in sorted(category_to_id.items(), key=lambda x: x[1]):
        print(f"   - Phase_GT={phase_id}: {phase_name}")
    
    # 显示前几行示例
    print("\n📋 前5行示例:")
    print(df_all.head().to_string())


if __name__ == "__main__":
    generate_csv()

