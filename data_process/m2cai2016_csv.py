import os
import pandas as pd
import glob
from pathlib import Path

PHASE_MAPPING = {
    "TrocarPlacement": 0,
    "Preparation": 1,
    "CalotTriangleDissection": 2,
    "ClippingCutting": 3,
    "GallbladderDissection": 4,
    "GallbladderPackaging": 5,  # 修正了原参考文件中的拼写
    "CleaningCoagulation": 6,
    "GallbladderRetraction": 7
}

# 读取视频标注文件(25 fps 标注)
## train:data/micai2016/train_dataset/{video_name}.txt
## test:data/micai2016/test_dataset/{video_name}.txt
## val:data/micai2016/test_dataset/{video_name}.txt

## 转换为 1 fps，并判断帧存不存在
### test frame dir：data/Surge_Frames/M2CAI2016/frames/{video_name}/{video_name}_{index:08d}.jpg
### test frame dir：data/Surge_Frames/M2CAI2016/frames/{video_name}/{video_name}_{index:08d}.jpg

## 将标注文件转换为 csv 文件命名为train_metadata.csv，test_metadata.csv，val_metadata.csv
### csv文件关键字：【index,Hospital,Year,Case_Name,Case_ID,Frame_Path,Phase_GT,Phase_Name,Split】，Hospital为m2cai2016，Year为2016，Case_Name为视频名，Case_ID为视频id，Frame_Path为帧路径，Phase_GT为标注的阶段，Phase_Name为标注的阶段名称，Split为训练集、验证集、测试集

def read_annotation_file(file_path):
    """读取标注文件"""
    annotations = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # 跳过标题行
            parts = line.strip().split()
            if len(parts) >= 2:
                frame_idx = int(parts[0])
                phase = parts[1]
                annotations.append((frame_idx, phase))
    return annotations

def convert_25fps_to_1fps(annotations):
    """将25fps的标注转换为1fps"""
    converted = {}
    for frame_25fps, phase in annotations:
        # 25fps到1fps的转换：每25帧取1帧
        frame_1fps = frame_25fps // 25
        # 使用字典确保每个1fps帧只保留一个记录
        converted[frame_1fps] = phase
    # 转换为列表并按帧索引排序
    return sorted(converted.items())

def check_frame_exists(video_name, frame_idx):
    """检查帧文件是否存在"""
    frame_path = f"data/Surge_Frames/M2CAI16/frames/{video_name}/{video_name}_{frame_idx:08d}.jpg"
    return os.path.exists(frame_path), frame_path

def collect_all_case_ids():
    """收集所有数据集中的唯一case_id"""
    all_case_ids = set()
    
    # 收集训练集和测试集中的所有case_id
    annotation_dirs = [
        'data/Landscopy/m2cai16/train_dataset',
        'data/Landscopy/m2cai16/test_dataset'
    ]
    
    for annotation_dir in annotation_dirs:
        if os.path.exists(annotation_dir):
            annotation_files = glob.glob(os.path.join(annotation_dir, "*.txt"))
            annotation_files = [f for f in annotation_files if not f.endswith('_timestamp.txt') and not f.endswith('_pred.txt')]
            
            for annotation_file in annotation_files:
                video_name = os.path.basename(annotation_file).replace('.txt', '')
                # 提取视频ID
                if video_name.startswith('test_'):
                    case_id = video_name.replace('test_', '')
                else:
                    case_id = video_name
                all_case_ids.add(case_id)
    
    return sorted(list(all_case_ids))

def create_case_id_mapping(case_ids):
    """创建从字符串case_id到数字的映射"""
    return {case_id: idx for idx, case_id in enumerate(case_ids)}

def process_dataset(split_name, annotation_dir, output_csv, case_id_mapping):
    """处理数据集并生成CSV文件"""
    all_data = []
    
    # 获取所有标注文件
    annotation_files = glob.glob(os.path.join(annotation_dir, "*.txt"))
    annotation_files = [f for f in annotation_files if not f.endswith('_timestamp.txt') and not f.endswith('_pred.txt')]
    
    print(f"处理 {split_name} 数据集，找到 {len(annotation_files)} 个标注文件")
    
    for annotation_file in annotation_files:
        video_name = os.path.basename(annotation_file).replace('.txt', '')
        
        # 提取视频ID（字符串形式）
        if video_name.startswith('test_'):
            case_id_str = video_name.replace('test_', '')
        else:
            case_id_str = video_name
        
        # 使用映射将字符串case_id转换为数字
        case_id = case_id_mapping.get(case_id_str, -1)
        if case_id == -1:
            print(f"警告: case_id {case_id_str} 不在映射表中")
            
        print(f"处理视频: {video_name} (case_id: {case_id_str} -> {case_id})")
        
        # 读取标注文件
        annotations = read_annotation_file(annotation_file)
        
        # 转换为1fps
        converted_annotations = convert_25fps_to_1fps(annotations)
        
        # 处理每一帧
        for frame_1fps, phase in converted_annotations:
            # 检查帧文件是否存在
            frame_exists, frame_path = check_frame_exists(video_name, frame_1fps + 1)  # +1因为帧索引从1开始
            
            if frame_exists:
                data = {
                    'index': frame_1fps,
                    'Hospital': 'm2cai2016',
                    'Year': 2016,
                    'Case_Name': video_name,
                    'Case_ID': case_id,  # 现在使用数字ID
                    'Frame_Path': frame_path,
                    'Phase_GT': PHASE_MAPPING.get(phase, -1),
                    'Phase_Name': phase,
                    'Split': split_name
                }
                all_data.append(data)
            else:
                print(f"警告: 帧文件不存在 {frame_path}")
    
    # 创建DataFrame并保存
    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False)
    print(f"已保存 {split_name} 数据集到 {output_csv}，共 {len(df)} 条记录")
    
    return df

def main():
    """主函数"""
    print("开始处理M2CAI2016数据集...")
    
    # 收集所有唯一的case_id并创建映射
    print("收集所有case_id...")
    all_case_ids = collect_all_case_ids()
    case_id_mapping = create_case_id_mapping(all_case_ids)
    print(f"找到 {len(all_case_ids)} 个唯一的case_id")
    print(f"Case_ID映射示例: {list(case_id_mapping.items())[:5]}")  # 显示前5个映射
    
    # 处理训练集
    train_df = process_dataset(
        'train',
        'data/Landscopy/m2cai16/train_dataset',
        'data/Surge_Frames/M2CAI16/train_metadata.csv',
        case_id_mapping
    )
    
    # 处理测试集
    test_df = process_dataset(
        'test',
        'data/Landscopy/m2cai16/test_dataset',
        'data/Surge_Frames/M2CAI16/test_metadata.csv',
        case_id_mapping
    )
    
    # 处理验证集（使用测试集数据）
    val_df = process_dataset(
        'val',
        'data/Landscopy/m2cai16/test_dataset',
        'data/Surge_Frames/M2CAI16/val_metadata.csv',
        case_id_mapping
    )
    
    print("数据集处理完成！")
    print(f"训练集: {len(train_df)} 条记录")
    print(f"测试集: {len(test_df)} 条记录")
    print(f"验证集: {len(val_df)} 条记录")

if __name__ == "__main__":
    main()


