import pandas as pd
import os
import shutil
import argparse

def add_index_label(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: 文件不存在: {csv_path}")
        return

    print(f"正在处理: {csv_path}")
    
    # 备份
    bak_path = csv_path + ".bak"
    if not os.path.exists(bak_path):
        try:
            shutil.copy(csv_path, bak_path)
            print(f"已备份原文件至: {bak_path}")
        except Exception as e:
            print(f"备份失败: {e}")
    else:
        print(f"备份文件已存在: {bak_path}，跳过备份")

    try:
        # 读取
        df = pd.read_csv(csv_path)
        
        # 添加 Index
        if 'Index' not in df.columns:
            print("添加 Index 列...")
            # 插入到第一列
            df.insert(0, 'Index', range(len(df)))
        else:
            print("Index 列已存在，跳过添加。")
            
        # 添加 label
        if 'label' not in df.columns:
            print("添加 label 列...")
            df['label'] = -1
        else:
            print("label 列已存在，跳过添加。")
        
        # 保存
        df.to_csv(csv_path, index=False)
        print(f"完成保存: {csv_path}")
        
    except Exception as e:
        print(f"处理文件时出错: {e}")

if __name__ == "__main__":
    # 默认路径
    default_path = "/home/jinlin_wu/NSJepa/data/Surge_Frames/EndoFM_Private/clips_16f/unlabeled_dense_16f_detailed.csv"
    
    parser = argparse.ArgumentParser(description="给CSV文件添加Index和label列")
    parser.add_argument("csv_path", nargs="?", default=default_path, help="CSV文件路径")
    
    args = parser.parse_args()
    
    add_index_label(args.csv_path)

