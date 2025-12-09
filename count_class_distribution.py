import pandas as pd
import os

files = [
    'data/Surge_Frames/AIxsuture_v1/test_clip_metadata.csv',
    'data/Surge_Frames/AIxsuture_v1/train_clip_metadata.csv',
    'data/Surge_Frames/AIxsuture_v1/val_clip_metadata.csv'
]

def count_classes(files):
    total_stats = {}
    
    for file_path in files:
        if os.path.exists(file_path):
            print(f"--- 统计文件: {file_path} ---")
            try:
                df = pd.read_csv(file_path)
                
                # 检查列名
                target_col = None
                if 'label_name' in df.columns:
                    target_col = 'label_name'
                elif 'label' in df.columns:
                    target_col = 'label'
                
                if target_col:
                    counts = df[target_col].value_counts().sort_index()
                    print(f"按 '{target_col}' 统计数量:")
                    print(counts)
                    print(f"总计行数: {len(df)}")
                    
                    # 汇总统计
                    for label, count in counts.items():
                        total_stats[label] = total_stats.get(label, 0) + count
                else:
                    print("未找到 'label_name' 或 'label' 列。")
                    print(f"现有列名: {df.columns.tolist()}")

            except Exception as e:
                print(f"读取文件出错 {file_path}: {e}")
            print("\n")
        else:
            print(f"文件不存在: {file_path}\n")

    print("--- 所有文件汇总统计 ---")
    if total_stats:
        for label, count in sorted(total_stats.items()):
            print(f"{label}: {count}")
        print(f"总计: {sum(total_stats.values())}")
    else:
        print("无数据汇总。")

if __name__ == "__main__":
    count_classes(files)

