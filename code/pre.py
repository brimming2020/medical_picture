import numpy as np
import cv2


def pre(img_path):
    # img_path = "lab\M102.png"
    img_ = cv2.imread(img_path, flags=0)
    # cv2.imshow('0',img_)
    _, img_binary = cv2.threshold(img_, 100, 0, cv2.THRESH_TOZERO_INV)
    # print (img_binary)
    _, img_binary = cv2.threshold(img_binary, 1, 225, cv2.THRESH_BINARY)
    # cv2.imshow('1',img_binary)
    kernel = np.ones((2, 2), np.uint8)
    kernel1 = np.ones((5, 5), np.uint8)
    kernel2 = np.ones((40, 40), np.uint8)
    closing = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)

    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel1)
    opening = cv2.dilate(opening, kernel2, 25)
    for i in range(0, 512):
        for j in range(0, 512):
            if (opening[i, j] != 225):
                img_[i, j] = 0
    # cv2.imshow('2',img_)

    # cv2.imshow('img1',img_binary)
    # img_binary=bone_extract2(img_binary)
    # cv2.imshow('3',opening)
    return img_
