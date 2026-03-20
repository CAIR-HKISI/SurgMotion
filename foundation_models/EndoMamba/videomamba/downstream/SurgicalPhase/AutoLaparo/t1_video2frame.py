import cv2
import os


video_src_path = "/mnt/tqy/AutoLaparo/AutoLaparo_Task1/videos/"
frame_save_path = "/mnt/tqy/AutoLaparo/AutoLaparo_Task1/videos_extracted/"
if not os.path.exists(frame_save_path):
    os.mkdir(frame_save_path)


def img_cut(image):
    binary_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image2 = cv2.threshold(binary_image, 15, 255, cv2.THRESH_BINARY)
    binary_image2 = cv2.medianBlur(binary_image2, 19)
    x = binary_image2.shape[0]
    y = binary_image2.shape[1]

    edges_x = []
    edges_y = []
    for i in range(x):
        for j in range(y):
            if binary_image2.item(i, j) != 0:
                edges_x.append(i)
                edges_y.append(j)

    if not edges_x:
        return image

    left = min(edges_y)  # left border
    right = max(edges_y)
    bottom = min(edges_x)
    top = max(edges_x)

    img1 = image[bottom:top, left:right]
    return img1


video_path = sorted(os.listdir(video_src_path))
for videos in video_path:

    video_path = video_src_path + videos
    video_name = videos[:2]
    if not os.path.exists(frame_save_path + video_name):
        os.mkdir(frame_save_path + video_name)

    cap = cv2.VideoCapture(video_path)
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % 25 == 0:    # down_sample from 25 to 1 fps
            img_save_path = frame_save_path + video_name + '/' + str(frame_num//25 + 1).zfill(4) + ".jpg"

            dim = (int(frame.shape[1]/frame.shape[0]*300), 300)
            frame = cv2.resize(frame, dim, cv2.INTER_AREA)
            frame_no_black = img_cut(frame)
            img_result = cv2.resize(frame_no_black, (250, 250), cv2.INTER_AREA)

            cv2.imwrite(img_save_path, img_result)
            # cv2.waitKey(1)

        frame_num += 1

    cap.release()
    print("Video {:s}: Totally have {:d} frames".format(video_name, frame_num))

print("Done")


