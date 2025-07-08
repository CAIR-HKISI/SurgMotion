import os
import cv2
import csv

def split_videos(input_dir, output_dir, clip_duration=3):
    """
    将input_dir下所有mp4视频按clip_duration（秒）切割，保存到output_dir
    返回每个切割后片段的完整路径、文件名、标签（-1）
    """
    os.makedirs(output_dir, exist_ok=True)
    video_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
    split_info = []

    for video_file in video_files:
        video_path = os.path.join(input_dir, video_file)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            print(f"无法获取FPS: {video_path}")
            cap.release()
            continue

        frames_per_clip = int(clip_duration * fps)
        basename = os.path.splitext(video_file)[0]

        clip_idx = 0
        frame_idx = 0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        while frame_idx < total_frames:
            out_name = f"{basename}_clip{clip_idx:04d}.mp4"
            out_path = os.path.join(output_dir, out_name)
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            frames_written = 0
            while frames_written < frames_per_clip and frame_idx < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                frames_written += 1
                frame_idx += 1

            out.release()
            split_info.append((out_path, out_name, -1))
            clip_idx += 1

        cap.release()
        print(f"{video_file} 切割完成，共生成 {clip_idx} 个片段。")
    return split_info

def save_csv(split_info, csv_path):
    """
    保存切割后视频的信息到csv文件
    """
    with open(csv_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["path", "name", "label"])
        for row in split_info:
            writer.writerow(row)
    print(f"CSV文件已保存到: {csv_path}")



if __name__ == "__main__":
    input_dir = "/data2/wjl/pitvis/"
    output_dir = "/data2/wjl/pitvis_split_3s/"
    clip_duration = 3
    csv_path = os.path.join(output_dir, "split_videos.csv")

    split_info = split_videos(input_dir, output_dir, clip_duration)
    save_csv(split_info, csv_path)
