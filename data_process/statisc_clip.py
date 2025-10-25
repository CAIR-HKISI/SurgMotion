import os
import glob

# 根目录（根据你的路径修改）
root_dir = "data/Surge_Frames"

# 匹配所有 clips_64f 目录中的 txt 文件
pattern = os.path.join(root_dir, "**", "clips_64f", "**", "*.txt")

# 获取所有匹配的文件列表
all_txt_files = glob.glob(pattern, recursive=True)

# 用于分类汇总（按数据集名称）
dataset_counts = {}

for file_path in all_txt_files:
    # 获取相对路径
    rel_path = os.path.relpath(file_path, root_dir)
    
    # 数据集名称（即第一层文件夹，比如 "PolypDiag"）
    dataset_name = rel_path.split(os.sep)[0]
    
    dataset_counts[dataset_name] = dataset_counts.get(dataset_name, 0) + 1

# 输出结果
print(f"{'Dataset':<40} | {'64f clips count':>15}")
print("-" * 60)
for dataset, count in sorted(dataset_counts.items()):
    print(f"{dataset:<40} | {count:>15}")

# 总和
print("-" * 60)
print(f"{'Total':<40} | {len(all_txt_files):>15}")
