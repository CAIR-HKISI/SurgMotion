import cv2
import os
import numpy as np
import concurrent.futures
from tqdm import tqdm
import threading  # 导入threading模块以使用Lock

def is_uniform_frame(frame, std_threshold=1):
    """
    判断一帧图像是否为异常帧（即颜色变化非常小，标准差非常低）
    """
    # 将图像转换为灰度图像来简化计算（可以根据需要选择其他颜色空间）
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 计算标准差
    std_dev = np.std(gray_frame)

    # 如果标准差小于阈值，则认为是异常帧
    return std_dev < std_threshold


def check_video_for_normal_frames(video_path):
    """
    检查视频中的所有帧是否都正常
    返回正常的帧数，如果包含异常帧，则返回-1
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return -1

    total_frames = 0
    normal_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        if not is_uniform_frame(frame):
            normal_frames += 1
        else:
            return -1

    cap.release()

    # 如果没有任何异常帧，返回正常帧数
    if normal_frames == total_frames:
        return total_frames
    else:
        return -1

def process_video(video_path, directory_path, train_list, lock, std_threshold):
    """
    处理单个视频，检查异常帧，返回处理结果
    """
    normal_frames = check_video_for_normal_frames(video_path)

    if normal_frames != -1:
        # 视频没有异常帧，添加到训练集
        with lock:
            filename = video_path.split('/')[-1]
            root = video_path.split(filename)[:-1]
        if ' ' in filename:
            old_file_path = os.path.join(root, filename)
            new_filename = filename.replace(' ', '')
            new_file_path = os.path.join(root, new_filename)

            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {old_file_path} -> {new_file_path}")
            
            filename = new_filename
        
        train_list.append(filename)
        return normal_frames

    else:
        # 重命名为 (ignore)
        new_filename = video_path.replace('.', '(ignore).', 1)  # 在文件名的第一个点前添加 (ignore)
        new_video_path = os.path.join(directory_path, new_filename)
        os.rename(video_path, new_video_path)
        print(f"{video_path} abnormal")
        return 0

def process_directory(directory_path, std_threshold=1.0):
    """
    遍历目录，检查每个视频，保存正常视频并统计帧数
    使用多线程并显示进度条
    """
    train_list = []
    total_video_frames = 0
    video_count = 0

    # 获取所有视频文件路径
    video_files = [os.path.join(directory_path, filename) for filename in os.listdir(directory_path)
                   if filename.endswith(('.mp4', '.avi', '.mov', '.mkv'))]  # 支持的视频格式
    
    video_files.sort()

    # 设置线程池，并行处理视频
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 创建一个共享的锁来保证对`train_list`的线程安全访问
        lock = threading.Lock()

        # 使用tqdm显示进度条
        for normal_frames in tqdm(executor.map(lambda video_path: process_video(video_path, directory_path, train_list, lock, std_threshold), video_files),
                                  total=len(video_files), desc="Processing Videos"):
            if normal_frames != 0:
                total_video_frames += normal_frames
                video_count += 1

    # 将正常视频路径写入train_list.txt
    with open('/mnt/tqy/ROBUST-MIS/train_list.txt', 'w') as f:
        for video_path in train_list:
            line = f"{video_path} 0\n"
            f.write(line)

    print(f"正常视频总数: {video_count}")
    print(f"所有正常视频帧数总计: {total_video_frames}")


# 使用示例：指定视频所在目录
directory_path = '/mnt/tqy/ROBUST-MIS/'
process_directory(directory_path)
