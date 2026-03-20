import os
import cv2
import re
import numpy as np


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]



def picvideo(path, size=(256, 256), extension='.png', fps=10, fcc='mp4v',
             isFlip=False, flipType=-1, dir=None):
    
    filelist = [f for f in os.listdir(path) if f.endswith(extension)]
    filelist.sort(key=natural_sort_key)

    file_path = dir + ".mp4"
    fourcc = cv2.VideoWriter_fourcc(*fcc)

    video = cv2.VideoWriter(file_path, fourcc, fps, size)

    for item in filelist:
        item = path + '/' + item
        img = cv2.imread(item)
        if size is None:
            size = (np.shape(img)[0], np.shape(img)[1])
            video = cv2.VideoWriter(file_path, fourcc, fps, size)
        img = cv2.resize(img, size)
        if isFlip:
            img = cv2.flip(img, flipType)
        # cv2.imshow('video', img)
        # cv2.waitKey(1)
        video.write(img)  

    video.release() 
    cv2.waitKey(1) & 0xFF
    cv2.destroyAllWindows()
    
    print("Generated video ", file_path)


if __name__ == '__main__':
    base_dir = '/mnt/tqy/SUN/SUN-Negative/'
    dirs = os.listdir(base_dir)
    save_dir = '/mnt/tqy/SUN/videos/'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for d in dirs:
        picvideo(base_dir + d, None, fps=30, extension='.jpg', dir=save_dir + d)
