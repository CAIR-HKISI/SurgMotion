
"""
csv文件中，clip_id是字符不是数字，导致下游处理出现问题。
功能：扫描 data/Surge_Frames/AVOS/clips_64f 目录下所有csv文件（通常以 train/val/test 命名），
将所有clip_id（或case_id）字符串建立唯一编号映射（从0开始），
统一替换到所有csv文件中，并覆盖保存。
"""

import os
import pandas as pd
from glob import glob

import argparse

def main():
    parser = argparse.ArgumentParser(description="统一csv文件中clip_id/case_id的编号")
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="包含待处理csv文件的目录路径"
    )
    args = parser.parse_args()
    base_dir = args.base_dir

    # 搜索所有csv文件（假设全是我们需要处理的）
    csv_files = glob(os.path.join(base_dir, "*.csv"))
    
    # 第一次收集所有出现过的clip_id（或case_id，对应列名）
    all_ids = set()
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        # 兼容 clip_id 或 case_id 命名
        if 'clip_id' in df.columns:
            ids = df['clip_id'].astype(str)
        elif 'case_id' in df.columns:
            ids = df['case_id'].astype(str)
        else:
            raise ValueError(f"{csv_path} 缺少clip_id或case_id列")
        all_ids.update(ids)
    all_ids = sorted(list(all_ids))
    id_map = {id_str: idx for idx, id_str in enumerate(all_ids)}
    print(f"共{len(id_map)}个clip_id/case_id被统一编号。")

    # 第二次写回
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        id_col = 'clip_id' if 'clip_id' in df.columns else 'case_id'
        df[id_col] = df[id_col].astype(str).map(id_map)
        df.to_csv(csv_path, index=False)
        print(f"已处理覆盖: {csv_path}")

if __name__ == "__main__":
    main()



