
"""
csv文件中，没有index列，导致后续处理出现问题
功能：扫描 data/Surge_Frames/AVOS/clips_64f 目录下所有csv文件（通常以 train/val/test 命名），
将所有csv所有行第一列（或无论有无index/index/Index）赋予新的唯一编号（从0开始，命名为index），
统一替换到所有csv文件中，并覆盖保存。
"""

import pandas as pd
import os
import argparse
import glob

def fix_index_id(csv_path):
    df = pd.read_csv(csv_path)
    columns = list(df.columns)
    columns_lower = [col.lower() for col in columns]
    if "index" not in columns_lower:
        # 没有Index/index列，添加Index列
        df.insert(0, "Index", range(len(df)))
    else:
        # 有index/Index列，统一替换成Index
        idx = columns_lower.index("index")
        cols_before = columns[:idx]
        cols_after = columns[idx+1:]
        # 去掉原有的index列
        df_no_idx = df.drop(columns=[columns[idx]])
        # 在原位置插入Index列
        df_no_idx.insert(idx, "Index", range(len(df)))
        df = df_no_idx
    df.to_csv(csv_path, index=False)
    return df

def main():
    parser = argparse.ArgumentParser(description="统一csv文件中Index列的编号")
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="包含待处理csv文件的目录路径"
    )
    args = parser.parse_args()
    base_dir = args.base_dir

    csv_files = glob.glob(os.path.join(base_dir, "*.csv"))
    for csv_file in csv_files:
        try:
            fix_index_id(csv_file)
        except Exception as e:
            print(f"处理文件 {csv_file} 时出错: {e}")

if __name__ == "__main__":
    main()
