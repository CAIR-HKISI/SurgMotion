import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置视频文件夹路径
video_dir = "/data/wjl/vjepa2/data/ssv2/20bn-something-something-v2/"

# 获取所有视频文件名（假设为mp4格式）
video_files = [f for f in os.listdir(video_dir) if f.endswith('.webm')]
print(f"总共找到 {len(video_files)} 个视频文件。")

# 随机抽取1000个视频
sample_num = 1000
if len(video_files) < sample_num:
    print(f"视频数量不足{sample_num}，实际只抽取{len(video_files)}个。")
    sample_num = len(video_files)
sampled_files = random.sample(video_files, sample_num)

# 统计每个视频的时长（秒）
durations = []
for idx, vf in enumerate(sampled_files):
    video_path = os.path.join(video_dir, vf)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        continue
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if fps > 0:
        duration = frame_count / fps
    else:
        duration = 0
    durations.append(duration)
    cap.release()
    if (idx+1) % 100 == 0:
        print(f"已处理 {idx+1}/{sample_num} 个视频。")

# print 平均时长，最大时长，最小时长，方差，中位数，单位为秒
print(f"平均时长: {np.mean(durations):.2f} 秒")
print(f"最大时长: {np.max(durations):.2f} 秒")
print(f"最小时长: {np.min(durations):.2f} 秒")
print(f"方差: {np.var(durations):.2f} (秒^2)")
print(f"中位数: {np.median(durations):.2f} 秒")

# Plot the histogram of video durations (handle the case when durations is empty)
if len(durations) == 0:
    print("No available video duration data, cannot plot histogram.")
else:
    plt.figure(figsize=(10, 6))
    plt.hist(durations, bins=50, color='skyblue', edgecolor='black')
    plt.xlabel('Video Duration (seconds)')
    plt.ylabel('Number of Videos')
    plt.title(f'Duration Distribution of {sample_num} Randomly Sampled SSV2 Videos')
    plt.grid(True)
    plt.savefig("ssv2_duration_distribution.png")



