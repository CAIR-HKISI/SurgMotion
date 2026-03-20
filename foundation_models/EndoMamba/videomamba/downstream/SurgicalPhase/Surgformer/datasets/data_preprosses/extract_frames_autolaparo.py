# import numpy as np
# import os
# import cv2
# from tqdm import tqdm

# ROOT_DIR = "/mnt/tqy/AutoLaparo/AutoLaparo_Task1/"
# VIDEO_NAMES = os.listdir(os.path.join(ROOT_DIR, "videos"))
# VIDEO_NAMES = sorted([x for x in VIDEO_NAMES if 'mp4' in x])

# FRAME_NUMBERS = 0

# for video_name in VIDEO_NAMES:
#     print(video_name)
#     vidcap = cv2.VideoCapture(os.path.join(ROOT_DIR, "videos", video_name))
#     fps = vidcap.get(cv2.CAP_PROP_FPS)
#     print("fps", fps)
#     success=True
#     count=0
#     save_dir = '/mnt/tqy/AutoLaparo/AutoLaparo_Task1/frames/' + video_name.replace('.mp4', '') +'/'
#     save_dir = os.path.join(ROOT_DIR, save_dir)
#     os.makedirs(save_dir, exist_ok=True)
#     while success is True:
#         success,image = vidcap.read()
#         if success:
#             if count % fps == 0:
#                 cv2.imwrite(save_dir + str(int(count//fps)).zfill(5) + '.png', image)
#             count+=1
#     vidcap.release()
#     cv2.destroyAllWindows()
#     print(count)
#     FRAME_NUMBERS += count

# print('Total Frams', FRAME_NUMBERS)

import numpy as np
import os
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT_DIR = "/mnt/tqy/AutoLaparo/AutoLaparo_Task1/"
VIDEO_NAMES = os.listdir(os.path.join(ROOT_DIR, "videos"))
VIDEO_NAMES = sorted([x for x in VIDEO_NAMES if 'mp4' in x])

def process_video(video_name):
    """
    Process each video to extract frames and save them as PNG images.
    """
    video_path = os.path.join(ROOT_DIR, "videos", video_name)
    save_dir = os.path.join(ROOT_DIR, "frames", video_name.replace('.mp4', ''))
    os.makedirs(save_dir, exist_ok=True)

    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    print(f"Processing {video_name} at {fps} FPS")

    success = True
    count = 0
    frame_numbers = 0

    while success:
        success, image = vidcap.read()
        if success:
            if count % fps == 0:  # Save every frame based on fps
                frame_filename = os.path.join(save_dir, f"{int(count // fps):05d}.png")
                cv2.imwrite(frame_filename, image)
                frame_numbers += 1
            count += 1

    vidcap.release()
    try:
        cv2.destroyAllWindows()
    except:
        pass
    print(f"Finished processing {video_name}, total frames: {frame_numbers}")
    return frame_numbers

def process_videos_parallel(video_names):
    """
    Use ProcessPoolExecutor to process videos in parallel.
    """
    total_frames = 0
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_video, video_name) for video_name in video_names]

        for future in as_completed(futures):
            total_frames += future.result()

    return total_frames

if __name__ == "__main__":
    total_frames = process_videos_parallel(VIDEO_NAMES)
    print(f"Total frames processed: {total_frames}")
