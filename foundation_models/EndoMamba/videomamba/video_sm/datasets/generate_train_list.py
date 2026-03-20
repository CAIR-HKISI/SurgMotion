import os

# 设置目标目录和输出文件路径
data_dir = "/mnt/tqy/GLENDA_v1.0/"
root_dir = "/mnt/tqy/GLENDA_v1.0/"
output_file = root_dir + "train_list.txt"

# 遍历目录并获取所有 .mp4 文件
def get_mp4_files(directory):
    mp4_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".mp4"):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, root_dir)
                mp4_files.append(relative_path)
    return mp4_files

# 获取所有 .mp4 文件的路径
mp4_files = get_mp4_files(data_dir)

# 写入 train_list.txt 文件，假设标签为 0
with open(output_file, "w") as f:
    for video_path in mp4_files:
        line = f"{video_path} 0\n"
        f.write(line)

print(f"Generated {output_file} with {len(mp4_files)} entries.")