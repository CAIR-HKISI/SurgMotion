import os
import pandas as pd

# 根目录
BASE_DIR = "data/Surge_Frames/AIxsuture/clips_64f"
# 三个文件名
CSV_FILES = [
    "train_dense_64f_detailed.csv",
    "val_dense_64f_detailed.csv",
    "test_dense_64f_detailed.csv",
]

# 标签列名，如果你的列名是 'label' 就改成 'label'
LABEL_COL = "Phase_GT"

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

    # 计算全局最小值和最大值
    all_values = pd.concat([df[LABEL_COL] for _, df in dfs], axis=0)
    vmin = all_values.min()
    vmax = all_values.max()

    print(f"全局 {LABEL_COL} 最小值: {vmin}")
    print(f"全局 {LABEL_COL} 最大值: {vmax}")

    if vmax == vmin:
        print("警告：最大值等于最小值，无法进行归一化。")
        return

    # 对每个文件做归一化，并保存新文件
    for fname, df in dfs:
        df[f"{LABEL_COL}_norm"] = (df[LABEL_COL] - vmin) / (vmax - vmin)
        out_name = fname.replace(".csv", "_norm.csv")
        out_path = os.path.join(BASE_DIR, out_name)
        df.to_csv(out_path, index=False)
        print(f"已保存归一化结果到: {out_path}")

if __name__ == "__main__":
    main()