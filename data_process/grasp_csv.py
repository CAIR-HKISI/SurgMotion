import os
import json
import pandas as pd
import re
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

# ========================================
# 基础路径配置
# ========================================
FRAME_PATH = "data/Surge_Frames/GraSP/frames"
TRAIN_ANNOT = "data/Landscopy/GraSP/GraSP_1fps/annotations/grasp_short-term_train.json"
TEST_ANNOT = "data/Landscopy/GraSP/GraSP_1fps/annotations/grasp_short-term_test.json"
OUT_DIR = "data/Surge_Frames/GraSP"

# 数据名称和年份
DATANAME = "GraSP"
YEAR = 2024

# 需要移除的动作类别（样本太少）
EXCLUDED_ACTIONS = ["Cut", "Pull", "Release", "Open Something", "Grasp"]

# 最小样本数阈值（低于此数量的类别将被移除）
MIN_SAMPLES_THRESHOLD = 100  # 可以根据实际情况调整


def extract_case_id(case_name):
    """
    从Case_Name（如 "CASE001"）提取数字ID（如 1）
    """
    # 提取数字部分
    match = re.search(r'\d+', case_name)
    if match:
        return int(match.group())
    return 0


def extract_frame_number(image_name):
    """
    从image_name（如 "CASE002/00001.jpg"）提取帧号（如 1）
    """
    frame_filename = image_name.split('/')[-1]  # 例如 "00001.jpg"
    frame_num_str = frame_filename.split('.')[0]  # 例如 "00001"
    return int(frame_num_str)


def process_annotations(json_path, split_name):
    """
    处理标注文件，提取每一帧的动作标签
    标注帧是动作的开始帧，从当前标注到下一个标注之间的所有帧都属于当前动作
    
    Args:
        json_path: JSON标注文件路径
        split_name: 数据集划分名称（"train" 或 "test"）
    
    Returns:
        DataFrame: 包含所有帧的标注信息
    """
    print(f"\n正在处理 {split_name} 集: {json_path}")
    
    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 创建动作ID到名称的映射
    action_id_to_name = {}
    for action_cat in data.get('actions_categories', []):
        action_id_to_name[action_cat['id']] = action_cat['name']
    
    # 按case_name分组，收集每个case的所有标注帧
    # 结构: {case_name: [(frame_num, action_id, action_name), ...]}
    case_annotations = defaultdict(list)
    
    # 遍历所有annotations，收集标注信息
    annotations = data.get('annotations', [])
    print(f"共有 {len(annotations)} 个标注对象")
    
    for ann in tqdm(annotations, desc=f"处理 {split_name} 标注"):
        image_id = ann['image_id']
        image_name = ann.get('image_name', f"image_{image_id}")
        
        # 解析case_name和frame_number
        case_name = image_name.split('/')[0]  # 例如 "CASE002"
        frame_num = extract_frame_number(image_name)
        
        # 收集动作标签（如果有多个动作，选择第一个）
        if 'actions' in ann:
            action_ids = ann['actions'] if isinstance(ann['actions'], list) else [ann['actions']]
            if action_ids:
                action_id = action_ids[0]  # 选择第一个动作
                action_name = action_id_to_name.get(action_id, "")
                case_annotations[case_name].append((frame_num, action_id, action_name))
    
    # 对每个case的标注按帧号排序，并去重（同一帧只保留一个标注）
    for case_name in case_annotations:
        # 使用字典去重，保留每个帧号第一次出现的标注
        unique_annotations = {}
        for frame_num, action_id, action_name in case_annotations[case_name]:
            if frame_num not in unique_annotations:
                unique_annotations[frame_num] = (frame_num, action_id, action_name)
        # 转换回列表并排序
        case_annotations[case_name] = sorted(unique_annotations.values(), key=lambda x: x[0])
    
    # 处理每个case，生成所有帧的标注
    all_data = []
    global_idx = 0
    frames_base = Path(FRAME_PATH)
    
    for case_name in tqdm(sorted(case_annotations.keys()), desc=f"生成 {split_name} CSV"):
        case_id = extract_case_id(case_name)
        case_dir = frames_base / case_name
        
        # 检查case目录是否存在
        if not case_dir.exists():
            print(f"⚠️  警告: 案例目录不存在: {case_dir}")
            continue
        
        # 获取该case的所有标注帧（已排序）
        annotations_list = case_annotations[case_name]
        
        if not annotations_list:
            continue
        
        # 获取该case目录下所有实际存在的帧文件
        frame_files = sorted([f for f in os.listdir(case_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        if not frame_files:
            print(f"⚠️  警告: 案例 {case_name} 没有帧文件")
            continue
        
        # 为每个实际存在的帧分配动作标签
        for frame_file in tqdm(frame_files, desc=f"处理 {case_name}", leave=False):
            # 提取帧号
            frame_num_str = frame_file.split('.')[0]
            try:
                frame_num = int(frame_num_str)
            except ValueError:
                continue
            
            # 找到该帧属于哪个标注区间
            # 标注帧是开始帧，从该帧到下一个标注帧之间的所有帧都属于该动作
            phase_gt = -1
            phase_name = "Unknown"
            
            # 如果当前帧在第一个标注帧之前，使用第一个标注的动作
            if frame_num < annotations_list[0][0]:
                phase_gt = annotations_list[0][1]
                phase_name = annotations_list[0][2]
            else:
                # 找到包含当前帧的标注区间
                for i, (annot_frame_num, action_id, action_name) in enumerate(annotations_list):
                    # 如果当前帧就是这个标注帧，或者在这个标注帧之后
                    if frame_num >= annot_frame_num:
                        # 检查是否是最后一个标注，或者当前帧在下一个标注之前
                        if i == len(annotations_list) - 1:
                            # 最后一个标注，之后的所有帧都属于这个动作
                            phase_gt = action_id
                            phase_name = action_name
                            break
                        else:
                            # 检查是否在下一个标注之前
                            next_annot_frame_num = annotations_list[i + 1][0]
                            if frame_num < next_annot_frame_num:
                                phase_gt = action_id
                                phase_name = action_name
                                break
            
            # 过滤掉需要移除的动作类别
            if phase_name in EXCLUDED_ACTIONS:
                continue
            
            # 构建帧路径
            frame_path = case_dir / frame_file
            frame_path_str = str(frame_path)
            
            # 检查文件是否存在（应该都存在，因为是从目录列表读取的）
            if not frame_path.exists():
                continue
            
            data_item = {
                'Index': global_idx,
                'DataName': DATANAME,
                'Year': YEAR,
                'Case_Name': case_name,
                'Case_ID': case_id,
                'Frame_Path': frame_path_str,
                'Phase_GT': phase_gt-1,
                'Phase_Name': phase_name,
                'Split': split_name
            }
            all_data.append(data_item)
            global_idx += 1
    
    df = pd.DataFrame(all_data)
    return df


def generate_csv():
    """生成CSV文件"""
    all_data = []
    
    # 处理训练集
    if os.path.exists(TRAIN_ANNOT):
        df_train = process_annotations(TRAIN_ANNOT, "train")
        all_data.append(df_train)
    else:
        print(f"⚠️  警告: 训练集标注文件不存在: {TRAIN_ANNOT}")
    
    # 处理测试集
    if os.path.exists(TEST_ANNOT):
        df_test = process_annotations(TEST_ANNOT, "test")
        all_data.append(df_test)
    else:
        print(f"⚠️  警告: 测试集标注文件不存在: {TEST_ANNOT}")
    
    if not all_data:
        print("❌ 错误: 没有找到任何标注文件")
        return
    
    # 合并所有数据
    df_all = pd.concat(all_data, ignore_index=True)
    
    # 重新分配Index
    df_all['Index'] = range(len(df_all))
    
    print(f"\n✅ 初始处理了 {len(df_all)} 帧（已过滤掉 {', '.join(EXCLUDED_ACTIONS)}）")
    
    # 统计初始分布
    print("\n" + "="*60)
    print("📊 初始 GT 分布统计")
    print("="*60)
    
    # 分别统计训练集和测试集
    df_train = df_all[df_all["Split"] == "train"]
    df_test = df_all[df_all["Split"] == "test"]
    
    train_actions = set(df_train['Phase_Name'].unique()) if not df_train.empty else set()
    test_actions = set(df_test['Phase_Name'].unique()) if not df_test.empty else set()
    
    print("\n📊 TRAIN 集初始 GT 分布:")
    if not df_train.empty:
        train_counts = df_train['Phase_Name'].value_counts().sort_values(ascending=False)
        total_train = len(df_train)
        for phase_name, count in train_counts.items():
            percentage = (count / total_train) * 100
            print(f"   - {phase_name}: {count} 帧 ({percentage:.2f}%)")
    
    print("\n📊 TEST 集初始 GT 分布:")
    if not df_test.empty:
        test_counts = df_test['Phase_Name'].value_counts().sort_values(ascending=False)
        total_test = len(df_test)
        for phase_name, count in test_counts.items():
            percentage = (count / total_test) * 100
            print(f"   - {phase_name}: {count} 帧 ({percentage:.2f}%)")
    
    # 找出只在训练集或只在测试集中出现的类别（没有重合）
    only_in_train = train_actions - test_actions
    only_in_test = test_actions - train_actions
    common_actions = train_actions & test_actions
    
    print(f"\n🔍 类别重合分析:")
    print(f"   - 训练集独有的类别: {sorted(only_in_train) if only_in_train else '无'}")
    print(f"   - 测试集独有的类别: {sorted(only_in_test) if only_in_test else '无'}")
    print(f"   - 训练集和测试集共有的类别: {sorted(common_actions) if common_actions else '无'}")
    
    # 找出样本数较少的类别
    all_action_counts = df_all['Phase_Name'].value_counts()
    low_sample_actions = set(all_action_counts[all_action_counts < MIN_SAMPLES_THRESHOLD].index)
    
    print(f"\n📉 样本数较少的类别 (< {MIN_SAMPLES_THRESHOLD} 帧):")
    for action in sorted(low_sample_actions):
        count = all_action_counts[action]
        print(f"   - {action}: {count} 帧")
    
    # 确定要移除的类别
    actions_to_remove = set()
    
    # 1. 移除只在训练集或只在测试集中出现的类别
    actions_to_remove.update(only_in_train)
    actions_to_remove.update(only_in_test)
    
    # 2. 移除样本数较少的类别
    actions_to_remove.update(low_sample_actions)
    
    # 3. 移除已经在EXCLUDED_ACTIONS中的类别
    actions_to_remove.update(EXCLUDED_ACTIONS)
    
    if actions_to_remove:
        print(f"\n🗑️  将移除以下类别: {sorted(actions_to_remove)}")
        # 过滤掉这些类别
        df_all = df_all[~df_all['Phase_Name'].isin(actions_to_remove)]
        # 重新分配Index
        df_all['Index'] = range(len(df_all))
    
    print(f"\n✅ 过滤后剩余 {len(df_all)} 帧")
    
    # 重新映射标签到0-N
    remaining_actions = sorted(df_all['Phase_Name'].unique())
    action_to_new_id = {action: idx for idx, action in enumerate(remaining_actions)}
    
    print(f"\n🔄 标签重新映射 (共 {len(remaining_actions)} 个类别):")
    for action, new_id in action_to_new_id.items():
        print(f"   - {action} -> {new_id}")
    
    # 更新Phase_GT为新的标签值
    df_all['Phase_GT'] = df_all['Phase_Name'].map(action_to_new_id)
    
    # 创建输出目录
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 分别保存train和test，并统计最终GT分布
    print("\n" + "="*60)
    print("📊 最终 GT 分布统计")
    print("="*60)
    
    for split in ["train", "test"]:
        df_split = df_all[df_all["Split"] == split]
        if not df_split.empty:
            out_csv = os.path.join(OUT_DIR, f"{split}_metadata.csv")
            df_split.to_csv(out_csv, index=False)
            print(f"\n💾 已保存 {len(df_split)} 帧到 {out_csv}")
            
            # 统计该split的GT分布
            print(f"\n📊 {split.upper()} 集最终 GT 分布:")
            # 按Phase_GT排序显示
            phase_gt_counts = df_split['Phase_GT'].value_counts().sort_index()
            total_frames = len(df_split)
            for phase_gt in sorted(phase_gt_counts.index):
                phase_name = df_split[df_split['Phase_GT'] == phase_gt]['Phase_Name'].iloc[0]
                count = phase_gt_counts[phase_gt]
                percentage = (count / total_frames) * 100
                print(f"   - Phase_GT={phase_gt}: {phase_name}: {count} 帧 ({percentage:.2f}%)")
        else:
            print(f"⚠️  {split} 集没有数据")
    
    # 打印整体统计信息（按Phase_GT排序）
    print("\n📊 整体最终动作标签统计:")
    phase_gt_counts = df_all['Phase_GT'].value_counts().sort_index()
    total_frames = len(df_all)
    for phase_gt in sorted(phase_gt_counts.index):
        phase_name = df_all[df_all['Phase_GT'] == phase_gt]['Phase_Name'].iloc[0]
        count = phase_gt_counts[phase_gt]
        percentage = (count / total_frames) * 100
        print(f"   - Phase_GT={phase_gt}: {phase_name}: {count} 帧 ({percentage:.2f}%)")
    
    # 显示前几行示例
    print("\n📋 前5行示例:")
    print(df_all.head().to_string())


if __name__ == "__main__":
    generate_csv()

