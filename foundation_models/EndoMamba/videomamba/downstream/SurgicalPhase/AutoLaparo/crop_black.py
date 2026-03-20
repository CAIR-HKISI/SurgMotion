import cv2
import numpy as np


def black_crop(img):
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # invert gray image
    gray = 255 - gray

    # threshold
    thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]

    # invert thresh
    thresh = 255 - thresh

    # get contours (presumably just one around the nonzero pixels)
    # then crop it to bounding rectangle
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = []
    contours = contours[0] if len(contours) == 2 else contours[1]
    for j in range(len(contours)):
        area.append(cv2.contourArea(contours[j]))
        max_idx = np.argmax(area)
    cntr = contours[max_idx]
    return cv2.boundingRect(cntr)


# read image
# img = cv2.imread('/home/t/data/ZSYY/Case10/frames/1000.jpg')
#
# x, y, w, h = black_crop(img)
# print(x, y, w, h )
# crop = img[y:y + h, x:x + w]
#
# # save cropped image
# cv2.imwrite('gymnast_crop.png', crop)

