import pandas as pd
import os
from pathlib import Path

# 标签和阶段名映射
PHASE_MAPPING = {
    -1: 'operation_ended',
    1: 'nasal corridor creation',
    2: 'anterior sphenoidotomy', 
    3: 'septum displacement',
    4: 'sphenoid sinus clearance',
    5: 'sellotomy',
    6: 'durotomy',
    7: 'tumour excision',
    8: 'haemostasis',
    9: 'synthetic_graft_placement',
    10: 'fat graft placement',
    11: 'gasket seal construct',
    12: 'dural sealant',
    13: 'nasal packing',
    14: 'debris clearance'
}

# 需要过滤的标签
FILTERED_PHASES = [-1, 11, 13]

# 数据集划分
TRAIN_VIDEOS = ['01', '03', '04', '05', '07', '08', '09', '10', '11', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '25']
VAL_VIDEOS = ['02', '06', '12', '13', '24']
TEST_VIDEOS = ['02', '06', '12', '13', '24']

def read_annotation_file(video_id):
    """读取单个视频的标注文件"""
    annotation_path = f"data/NeuroSurgery/pitvits/26531686/annotations_{video_id}.csv"
    if not os.path.exists(annotation_path):
        print(f"警告: 标注文件 {annotation_path} 不存在")
        return None
    
    try:
        df = pd.read_csv(annotation_path)
        # 确保列名正确
        expected_columns = ['int_video', 'int_time', 'int_step', 'int_instrument1', 'int_instrument2']
        if not all(col in df.columns for col in expected_columns):
            print(f"警告: 标注文件 {annotation_path} 列名不匹配")
            return None
        return df
    except Exception as e:
        print(f"读取标注文件 {annotation_path} 时出错: {e}")
        return None

def get_frame_path(video_id, frame_index):
    """生成帧文件路径"""
    return f"data/Surge_Frames/PitVis/frames/video_{video_id}/video_{video_id}_{frame_index:08d}.jpg"

def process_video_annotations(video_id):
    """处理单个视频的标注数据"""
    df = read_annotation_file(video_id)
    if df is None:
        return []
    
    processed_data = []
    
    for _, row in df.iterrows():
        int_step = row['int_step']
        
        # 过滤不需要的标签
        if int_step in FILTERED_PHASES:
            continue
        
        int_time = row['int_time']
        # 将秒转换为帧索引 (1fps, 所以1秒=1帧，帧编号从1开始)
        frame_index = int_time + 1
        
        # 检查帧文件是否存在
        frame_path = get_frame_path(video_id, frame_index)
        if not os.path.exists(frame_path):
            print(f"警告: 帧文件不存在 {frame_path}")
            continue
        
        # 获取阶段名称
        phase_name = PHASE_MAPPING.get(int_step, f"unknown_{int_step}")
        
        processed_data.append({
            'index': frame_index,
            'DataName': 'PitVis',
            'Year': 2023,
            'Case_Name': f"video_{video_id}",
            'Case_ID': video_id,
            'Frame_Path': frame_path,
            'Phase_GT': int_step,
            'Phase_Name': phase_name
        })
    
    return processed_data

def create_dataset_split(video_ids, split_name):
    """创建数据集划分"""
    all_data = []
    
    for video_id in video_ids:
        print(f"处理视频 {video_id}...")
        video_data = process_video_annotations(video_id)
        for item in video_data:
            item['Split'] = split_name
        all_data.extend(video_data)
    
    return all_data

def save_metadata_csv(data, filename):
    """保存元数据到CSV文件"""
    if not data:
        print(f"警告: {filename} 没有数据")
        return
    
    df = pd.DataFrame(data)
    
    # 确保输出目录存在
    output_dir = Path("data/Surge_Frames/PitVis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / filename
    df.to_csv(output_path, index=False)
    print(f"已保存 {output_path}, 共 {len(df)} 条记录")

def main():
    """主函数"""
    print("开始处理Pitvis数据集...")
    
    # 处理训练集
    print("\n处理训练集...")
    train_data = create_dataset_split(TRAIN_VIDEOS, 'train')
    save_metadata_csv(train_data, 'train_metadata.csv')
    
    # 处理验证集
    print("\n处理验证集...")
    val_data = create_dataset_split(VAL_VIDEOS, 'val')
    save_metadata_csv(val_data, 'val_metadata.csv')
    
    # 处理测试集
    print("\n处理测试集...")
    test_data = create_dataset_split(TEST_VIDEOS, 'test')
    save_metadata_csv(test_data, 'test_metadata.csv')
    
    print("\n处理完成!")
    
    # 打印统计信息
    print(f"\n统计信息:")
    print(f"训练集: {len(train_data)} 条记录")
    print(f"验证集: {len(val_data)} 条记录")
    print(f"测试集: {len(test_data)} 条记录")

if __name__ == "__main__":
    main()

