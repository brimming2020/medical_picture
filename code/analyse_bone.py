import pdb

import cv2
import os
import numpy as np
from multipolygon import isin_multipolygon


# 方法一，直接阈值分割
def bone_extract(img_binary):
    _, img_binary = cv2.threshold(img_binary, 127, 255, cv2.THRESH_BINARY)
    return img_binary


# 方法二，阈值分割后，用isin_multipolygon接口把骨骼内部的区域识别出来
def bone_extract2(img_binary):
    _, img_binary = cv2.threshold(img_binary, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((2, 2), np.uint8)
    img_open = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel)
    contours, hierarchy = cv2.findContours(img_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        points = contour[:, 0]
        most_top = np.min(points[:, 1])
        most_down = np.max(points[:, 1])
        most_left = np.min(points[:, 0])
        most_right = np.max(points[:, 0])
        points = points.tolist()

        for i in range(most_top, most_down + 1):
            for j in range(most_left, most_right + 1):
                if img_open[i, j] == 255:
                    continue

                if isin_multipolygon([j, i], points):
                    img_open[i, j] = 255
    return img_open


# 方法三，部分骨骼区域内部其实不需要识别，比如颅骨、脊柱，将这部分区域的给剔除掉
def bone_extract3(img_binary):
    _, img_binary = cv2.threshold(img_binary, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((2, 2), np.uint8)
    img_open = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel)

    _, horizontal = img_open.shape
    horizontal_left_border = int(horizontal / 3)
    horizontal_right_border = int(horizontal / 3 * 2)

    contours, hierarchy = cv2.findContours(img_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        points = contour[:, 0]
        most_top = np.min(points[:, 1])
        most_down = np.max(points[:, 1])
        most_left = np.min(points[:, 0])
        most_right = np.max(points[:, 0])
        points = points.tolist()

        for i in range(most_top, most_down + 1):
            for j in range(most_left, most_right + 1):
                if j < horizontal_right_border and j > horizontal_left_border:
                    continue

                if img_open[i, j] == 255:
                    continue

                if isin_multipolygon([j, i], points):
                    img_open[i, j] = 255
    return img_open


if __name__ == "__main__":
    img_path = "./data/a075.png"
    img_binary = cv2.imread(img_path, flags=0)
    _, img_binary = cv2.threshold(img_binary, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((2, 2), np.uint8)
    img_open = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel)
    img_rgb = cv2.cvtColor(img_open, cv2.COLOR_GRAY2BGR)

    cv2.imshow("img", img_open)
    contours, hierarchy = cv2.findContours(img_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    vertical, horizontal = img_open.shape
    horizontal_left_border = int(horizontal / 3)
    horizontal_right_border = int(horizontal / 3 * 2)
    # cv2.line(img_rgb,(int(horizontal/3),0),(int(horizontal/3),vertical),(0, 255, 0),thickness=3)
    # cv2.line(img_rgb,(int(horizontal/3*2),0),(int(horizontal/3*2),vertical),(0, 255, 0),thickness=3)

    for contour in contours:
        points = contour[:, 0]
        most_top = np.min(points[:, 1])
        most_down = np.max(points[:, 1])
        most_left = np.min(points[:, 0])
        most_right = np.max(points[:, 0])

        points = points.tolist()
        for i in range(most_top, most_down + 1):
            for j in range(most_left, most_right + 1):
                if horizontal_right_border > j > horizontal_left_border:
                    continue

                if img_open[i, j] == 255:
                    continue

                if isin_multipolygon([j, i], points):
                    img_open[i, j] = 255

    cv2.drawContours(img_rgb, contours, -1, (0, 0, 255), 2)
    cv2.imshow("img_rgb", img_rgb)
    cv2.imshow("img2", img_open)
    k = cv2.waitKey(0)
    if k == ord('s'):
        cv2.destroyAllWindows()
    # print(contours)
    # print(contours[0][:,0])
