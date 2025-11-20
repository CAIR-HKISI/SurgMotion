import os
import pandas as pd

# 根目录
BASE_DIR = "data/Surge_Frames/AIxsuture"
# 三个文件名
CSV_FILES = [
    "train_metadata.csv",
    "val_metadata.csv",
    "test_metadata.csv",
]

# 标签列名，如果你的列名是 'label' 就改成 'label'
LABEL_COL = "Phase_GT"

# 将连续分数量化为多少个整数类别（例如 5 档：0,1,2,3,4）
N_BINS = 5

def main():
    dfs = []
    # 先读取所有文件，检查列是否存在
    for fname in CSV_FILES:
        path = os.path.join(BASE_DIR, fname)
        if not os.path.isfile(path):
            print(f"文件不存在，跳过: {path}")
            continue
        df = pd.read_csv(path)
        if LABEL_COL not in df.columns:
            raise KeyError(f"{path} 中没有列 '{LABEL_COL}'，请检查列名。")
        dfs.append((fname, df))

    if not dfs:
        print("没有成功读取任何 CSV 文件。")
        return

    # 计算全局最小值和最大值（先归一化到 0~1 再量化）
    all_values = pd.concat([df[LABEL_COL] for _, df in dfs], axis=0)
    vmin = all_values.min()
    vmax = all_values.max()

    print(f"全局 {LABEL_COL} 最小值: {vmin}")
    print(f"全局 {LABEL_COL} 最大值: {vmax}")

    if vmax == vmin:
        print("警告：最大值等于最小值，无法进行归一化和量化。")
        return

    # 对每个文件做归一化 + 量化，并保存新文件
    for fname, df in dfs:
        # 归一化到 [0, 1]
        norm_col = f"{LABEL_COL}_norm"
        cls_col = f"{LABEL_COL}_cls"
        df[norm_col] = (df[LABEL_COL] - vmin) / (vmax - vmin)
        # 量化为 0 ~ (N_BINS-1) 的整数标签
        df[cls_col] = (df[norm_col] * (N_BINS - 1)).round().astype(int)
        df[cls_col] = df[cls_col].clip(0, N_BINS - 1)

        out_name = fname.replace(".csv", f"_quant{N_BINS}.csv")
        out_path = os.path.join(BASE_DIR, out_name)
        df.to_csv(out_path, index=False)
        print(f"已保存归一化 + 量化结果到: {out_path}")

if __name__ == "__main__":
    main()