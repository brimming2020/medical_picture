import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import math
import random
import csv


def my_cv_imread(filepath, flag=cv2.IMREAD_GRAYSCALE):
    # 使用imdecode函数进行读取
    img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), flag)
    return img


def my_cv_imwrite(path, img):
    cv2.imencode('.png', img)[1].tofile(path)


# 获取一个区域的边界,会过滤突然出现的小区域,保证获取到的是人体轮廓的外接矩阵
def get_frame_np(img_np):
    up = 0
    down = 0
    left = 0
    right = 0
    temp_up = np.zeros(img_np.shape[0])
    for i in range(img_np.shape[0]):
        flag = 0
        for j in range(img_np.shape[1]):
            if img_np[i][j] == 255:
                flag = 1
                break
        if flag == 1:
            temp_up[i] = 1
    num = 0
    for i in range(len(temp_up)):
        if num == 100:
            break
        if num == 0 and temp_up[i] == 1:
            up = i
            num += 1
        elif num != 0 and temp_up[i] == 1:
            if temp_up[i - 1] == 1:
                num += 1
            else:
                num = 0

    temp_down = np.zeros(img_np.shape[0])
    for i in range(img_np.shape[0] - 1, -1, -1):
        flag = 0
        for j in range(img_np.shape[1]):
            if img_np[i][j] == 255:
                flag = 1
                break
        if flag == 1:
            temp_down[i] = 1
    num = 0
    for i in range(img_np.shape[0] - 1, -1, -1):
        if num == 100:
            break
        if num == 0 and temp_down[i] == 1:
            down = i
            num += 1
        elif num != 0 and temp_down[i] == 1:
            if temp_down[i + 1] == 1:
                num += 1
            else:
                num = 0

    temp_left = np.zeros(img_np.shape[1])
    for j in range(img_np.shape[1]):
        flag = 0
        for i in range(img_np.shape[0]):
            if img_np[i][j] == 255:
                flag = 1
                break
        if flag == 1:
            temp_left[j] = 1
    num = 0
    for i in range(len(temp_left)):
        if num == 100:
            break
        if num == 0 and temp_left[i] == 1:
            left = i
            num += 1
        elif num != 0 and temp_left[i] == 1:
            if temp_left[i - 1] == 1:
                num += 1
            else:
                num = 0

    temp_right = np.zeros(img_np.shape[1])
    for j in range(img_np.shape[1] - 1, -1, -1):
        flag = 0
        for i in range(img_np.shape[0]):
            if img_np[i][j] == 255:
                flag = 1
                break
        if flag == 1:
            temp_right[j] = 1
    num = 0
    for i in range(img_np.shape[1] - 1, -1, -1):
        if num == 100:
            break
        if num == 0 and temp_right[i] == 1:
            right = i
            num += 1
        elif num != 0 and temp_right[i] == 1:
            if temp_right[i + 1] == 1:
                num += 1
            else:
                num = 0
    return np.array([left, right, up, down])


# 获取人体轮廓
def get_frame_body(img):
    # img = my_cv_imread(file_path)
    ret, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)

    # 开运算把相连的邻域分开
    kernel = np.ones((2, 2), np.uint8)
    img_open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((3, 3), np.uint8)
    img_close = cv2.dilate(img_open, kernel, iterations=1)

    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_close, connectivity=8)
    output_pre = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    # 过滤离散的小白点
    for i in range(1, num_labels):
        if stats[i][4] > 50:
            output_pre[i == labels] = 255
    contours, hierarch = cv2.findContours(output_pre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(output_pre, contours, -1, color=255, thickness=-1)
    # cv2.imshow('output', output)

    body_frame = get_frame_np(output_pre)
    # output_pre[body_frame[2], body_frame[0]:body_frame[1]] = 255
    # output_pre[body_frame[3], body_frame[0]:body_frame[1]] = 255
    # output_pre[body_frame[2]:body_frame[3], body_frame[0]] = 255
    # output_pre[body_frame[2]:body_frame[3], body_frame[1]] = 255
    return body_frame


def pre(img_path):
    # img_path = "lab\M102.png"
    img_ = my_cv_imread(img_path)
    cv2.imshow('0', img_)
    cv2.waitKey()
    _, img_binary = cv2.threshold(img_, 100, 0, cv2.THRESH_TOZERO_INV)
    cv2.imshow('1', img_binary)
    # cv2.waitKey()
    # print (img_binary)
    _, img_binary = cv2.threshold(img_binary, 1, 225, cv2.THRESH_BINARY)
    # cv2.imshow('1', img_binary)
    # cv2.waitKey()
    kernel = np.ones((2, 2), np.uint8)
    kernel1 = np.ones((5, 5), np.uint8)
    kernel2 = np.ones((40, 40), np.uint8)
    closing = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('2', closing)
    # cv2.waitKey()
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel1)
    # cv2.imshow('3', opening)
    # cv2.waitKey()
    opening = cv2.dilate(opening, kernel2, 25)
    # cv2.imshow('4', opening)
    cv2.waitKey()
    for i in range(0, 512):
        for j in range(0, 512):
            if opening[i, j] != 225:
                img_[i, j] = 0
    # cv2.imshow('2',img_)

    # cv2.imshow('img1',img_binary)
    # img_binary=bone_extract2(img_binary)
    # cv2.imshow('3',opening)
    return img_


# 低阈值的方法提取椎骨,仅提取椎骨
def find_max_region(img, area_key=100):
    # img = my_cv_imread(file_path)
    ret, binary = cv2.threshold(img, 75, 255, cv2.THRESH_BINARY)
    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    # 设置遮罩
    output = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    dis = [1e9]
    # 过滤离散的小白点
    for i in range(1, num_labels):
        if stats[i][4] > area_key:
            dis.append((img.shape[0] / 2 - centroids[i][0]) ** 2 + (img.shape[1] / 2 - centroids[i][1]) ** 2)
        else:
            dis.append(1e9)
    minS = np.argmin(dis)
    output[labels == minS] = 255
    kernel = np.ones((5, 5), np.uint8)
    output_close = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel)
    contours, hierarch = cv2.findContours(output_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(output_close, contours, -1, color=255, thickness=-1)

    # rett, binaryy = cv2.threshold(output, 75, 255, cv2.THRESH_BINARY_INV)
    # num_labels9, labels9, stats9, centroids9 = cv2.connectedComponentsWithStats(binaryy, connectivity=8)
    # d = [1e9, 1e9]
    # body_frame = get_frame_body(binaryy)
    # center = [(body_frame[0] + body_frame[1]) / 2, (body_frame[2] + body_frame[3]) / 2]
    # for i in range(2, num_labels9):
    #     if stats9[i][4] > 100:
    #         d.append((center[0] - centroids9[i][0]) ** 2 + (center[1] - centroids9[i][1]) ** 2)
    #     else:
    #         d.append(1e9)
    # minN = np.argmin(d)
    # output_close[labels9 == minN] = 0
    # kernel = np.ones((3, 3), np.uint8)
    # output_close = cv2.morphologyEx(output_close, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('close', output_close)
    return output_close


def find_nearest_region_in_pre(file_path, area_key=100):
    img = my_cv_imread(file_path)
    # cv2.imshow("pre", img)
    ret, binary = cv2.threshold(img, 75, 255, cv2.THRESH_BINARY)
    # cv2.imshow("binary", binary)
    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    # 设置遮罩
    output = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    dis = [1e9]
    # 过滤离散的小白点
    for i in range(1, num_labels):
        if stats[i][4] > area_key:
            dis.append((img.shape[0] / 2 - centroids[i][0]) ** 2 + (img.shape[1] / 2 - centroids[i][1]) ** 2)
        else:
            dis.append(1e9)
    min = np.argmin(dis)
    min_dis = dis[min]
    dis[min] = 1e9
    min2 = np.argmin(dis)

    for i in range(1, num_labels):
        mask = labels == i
        if i != min and i != min2:
            output[mask] = 0
        else:
            output[mask] = 255

    img[output != 255] = 0
    num_labels2, labels2, stats2, centroids2 = cv2.connectedComponentsWithStats(img, connectivity=8)
    # print(num_labels2)
    # cv2.imshow(file_path, img)
    # cv2.waitKey()
    if num_labels2 == 3:
        # print(stats[min][4], round(min_dis, 2), stats[min2][4], round(dis[min2], 2))
        return stats[min][4], round(min_dis, 2), stats[min2][4], round(dis[min2], 2)
    else:
        if num_labels2 == 2:
            return stats[min][4], round(min_dis, 2), -1, -1
        else:
            return -1, -1, -1, -1


def for_backbone(file_path, area_key=100):
    img = my_cv_imread(file_path)
    cv2.imshow("ori", img)
    ret, binary = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
    # cv2.imshow("binary", binary)
    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    output = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    # 过滤离散的小白点
    for i in range(1, num_labels):
        if stats[i][4] > 50:
            output[i == labels] = 255
    cv2.imshow("output", img)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(output, connectivity=8)

    img_np = np.array(output)
    frame = get_frame_np(img_np)
    center = [(frame[0] + frame[1]) / 2, (frame[2] + frame[3]) / 2]
    dis = [1e9]
    for i in range(1, num_labels):
        if stats[i][4] > area_key:
            dis.append((center[0] - centroids[i][0]) ** 2 + (center[1] - centroids[i][1]) ** 2)
        else:
            dis.append(1e9)
    min = np.argmin(dis)

    output2 = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    for i in range(1, num_labels):
        mask = labels == i
        if i != min:
            output2[mask] = 0
        else:
            output2[mask] = 255
    cv2.imshow('output2', output2)

    img_np = np.array(output2)
    frame = get_frame_np(img_np)
    center = [(frame[0] + frame[1]) / 2, (frame[2] + frame[3]) / 2]

    # 过滤离散的小白点
    # 椎骨中间那块黑黑的区域面积大概在100-400
    rett, binaryy = cv2.threshold(output2, 75, 255, cv2.THRESH_BINARY_INV)
    num_labels9, labels9, stats9, centroids9 = cv2.connectedComponentsWithStats(binaryy, connectivity=8)
    d = [1e9]
    for i in range(1, num_labels9):
        if 400 > stats9[i][4] > 100:
            d.append((center[0] - centroids9[i][0]) ** 2 + (center[1] - centroids9[i][1]) ** 2)
        else:
            d.append(1e9)
    min5 = np.argmin(d)
    output3 = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    if d[min5] != 1e9:
        for i in range(0, num_labels9):
            mask = labels9 == i
            if i != min5:
                output3[mask] = 0
            else:
                output3[mask] = 255
    cv2.imshow('output3', output3)

    contours, hierarch = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(output, contours, -1, color=255, thickness=-1)

    img2 = my_cv_imread(file_path, flag=cv2.IMREAD_COLOR)
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            if output[i, j] == 255:
                if output3[i, j] == 255:
                    img2[i, j] = [0, 0, 0]
                else:
                    img2[i, j] = [0, 255, 255]
    cv2.imshow('img', img2)
    cv2.waitKey()
    # my_cv_imwrite(
    #     'D:\\BenKe\\medical_picture\\三院\\三院\\code\\data\\' + os.path.basename(file_path).split('.')[
    #         0] + '-.png', img2)


def draw_frame(img, z_now, area_key=100, z_toal=238):
    # # 获取向前切片的z以及相对人体的位置
    # z = os.path.basename(file_path).split('.')[0][1:]
    # z = int(z[0]) * 100 + int(z[1]) * 10 + int(z[2])
    z_rate = z_now / z_toal

    ret, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    img2 = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    # 根据切片相对人体的位置进行不同的算法实现
    # 低于一定高度和高于一定高度时不存在显影剂,全部绘制
    if z_rate < 0.18:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        output = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        for i in range(1, num_labels):
            if stats[i][4] > 50:
                output[labels == i] = 255
        contours, hierarch = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img2, contours, -1, color=255, thickness=-1)
    elif z_rate > 0.5:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        output = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        for i in range(1, num_labels):
            if stats[i][4] > 50:
                output[labels == i] = 255
        # contours, hierarch = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # cv2.drawContours(img2, contours, -1, color=255, thickness=-1)
        img2 = output

    elif z_rate <= 0.5:
        body_frame = get_frame_body(img)
        body_dis = (body_frame[0] - body_frame[1]) ** 2 + (body_frame[2] - body_frame[3]) ** 2
        center = [(body_frame[0] + body_frame[1]) / 2, (body_frame[2] + body_frame[3]) / 2]

        # 开运算把相连的邻域分开
        kernel = np.ones((2, 2), np.uint8)
        img_open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((5, 5), np.uint8)
        img_close = cv2.dilate(img_open, kernel, iterations=1)
        # 连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_close, connectivity=8)
        output = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        # 过滤离散的小白点
        for i in range(1, num_labels):
            if stats[i][4] > 50:
                output[i == labels] = 255

        # 填充椎骨
        # 通过低阈值获取椎骨然后进而实现填充
        output_close = find_max_region(img, area_key)
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                if output_close[i, j] == 255 and output[i, j] != 255:
                    output[i, j] = 255

        cv2.imshow('output', output)
        cv2.waitKey()
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(output, connectivity=8)
        dis = [1e9]
        for i in range(1, num_labels):
            if stats[i][4] > area_key:
                dis.append((center[0] - centroids[i][0]) ** 2 + (center[1] - centroids[i][1]) ** 2)
            else:
                dis.append(1e9)
        minS = np.argmin(dis)

        output2 = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        # 不存在最近联通区则直接全部绘制
        if dis[minS] != 1e9:
            num = 0
            for i in range(1, num_labels):
                dis_temp = (centroids[i][0] - center[0]) ** 2 + (centroids[i][1] - center[1]) ** 2
                dis_relative = pow(dis_temp / body_dis, 0.5)
                if dis_relative < 0.45:
                    num += 1
            # 不存在离中心点较近的联通区则全部绘制
            if num > 1:
                # 优先考虑对称关系,骨骼才会出现高度对称的一对区域
                dis_x = [999]
                for i in range(1, num_labels):
                    if stats[i][4] > 1000:
                        dis_x.append(abs(centroids[i][0] - center[0]))
                    else:
                        dis_x.append(999)
                first_bone = []
                for i in range(1, num_labels):
                    if dis_x[i] != 999:
                        for j in range(i + 1, num_labels):
                            # 判断对称:x和y的坐标以及面积信息
                            # print(abs(dis_x[i] - dis_x[j]), abs(centroids[i][1] - centroids[j][1]), abs(
                            #         stats[i][4] - stats[j][4]))
                            # temp = np.zeros((img.shape[0], img.shape[1]), np.uint8)
                            # temp[labels == i] = 255
                            # temp[labels == j] = 255
                            # cv2.imshow('temp', temp)
                            # cv2.waitKey()
                            if abs(dis_x[i] - dis_x[j]) < 20 and abs(centroids[i][1] - centroids[j][1]) < 20 and abs(
                                    stats[i][4] - stats[j][4]) < 400:
                                first_bone.append(i)
                                first_bone.append(j)
                first_bone = np.array(first_bone)
                for i in range(1, num_labels):
                    # 判断是否在人体轮廓外(患者的手部)和是否处在对称的组中
                    if body_frame[0] < centroids[i][0] < body_frame[1] \
                            and body_frame[2] < centroids[i][1] < body_frame[3] and (i not in first_bone):
                        # 计算相对距离并判断是否在(0.04, 0.5)之间
                        dis_temp = (centroids[i][0] - center[0]) ** 2 + (centroids[i][1] - center[1]) ** 2
                        dis_relative = pow(dis_temp / body_dis, 0.5)
                        check_dis_relative = 0.04 < dis_relative < 0.35
                        # 计算x,y方向上的绝对距离和相对距离,判定y方向上在中心点以上 或 x方向上过度远离中心点
                        check_dis_xandy = ((centroids[i][1] < center[1] + 10) or abs(
                            (centroids[i][0] - center[0]) / (body_frame[0] - body_frame[1])) >= 0.35)
                        # 综合性地判断区域的距离
                        check_dis = check_dis_relative and check_dis_xandy
                        # 计算区域外接矩形的斜边长并判定是否大于8000
                        check_rec_hypotenuse = stats[i][2] ** 2 + stats[i][3] ** 2 <= 8000
                        # 判定位置信息:外接矩形的底边低于中心点或区域中心点在x方向上过于接近中心点
                        check_rec_pos = ((stats[i][1] + stats[i][3]) < center[1] + 20) or abs(
                            centroids[i][0] - center[0]) < 15
                        # 判定面积信息,查看区域外接矩形面积和区域面积的关系(显影剂分布很集中,呈椭圆形)
                        check_rec_area = stats[i][2] * stats[i][3] / stats[i][4] < 3
                        # 综合借助区域外接矩形进行判断
                        check_rec = check_rec_pos or check_rec_area
                        # 区域外接矩形过长是(>8000)才进行细致的判断
                        check_pos = check_rec_hypotenuse or ((not check_rec_hypotenuse) and check_rec)
                        if check_dis and check_pos:
                            # print('true:', dis_relative, stats[i][4], centroids[i][1], center[1] + 10,
                            #       round(abs((centroids[i][0] - center[0]) / (body_frame[0] - body_frame[1])), 2),
                            #       stats[i][2] ** 2 + stats[i][3] ** 2, stats[i][1] + stats[i][3],
                            #       centroids[i][0] - center[0], stats[i][2] * stats[i][3] / stats[i][4])
                            continue
                        else:
                            # print('false:', dis_relative, stats[i][4], centroids[i][1], center[1] + 10,
                            #       round(abs((centroids[i][0] - center[0]) / (body_frame[0] - body_frame[1])), 2),
                            #       stats[i][2] ** 2 + stats[i][3] ** 2, stats[i][1] + stats[i][3],
                            #       centroids[i][0] - center[0], stats[i][2] * stats[i][3] / stats[i][4])
                            output2[labels == i] = 255
                    else:
                        output2[labels == i] = 255
            else:
                output2 = output
        else:
            output2 = output
        contours, hierarch = cv2.findContours(output2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(output2, contours, -1, color=255, thickness=-1)
        # cv2.imshow('output2', output2)
        contours, hierarch = cv2.findContours(output2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img2, contours, -1, color=255, thickness=-1)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img2[i, j] == 255:
                img2[i, j] = img[i, j]
    cv2.imshow("frame", img2)
    cv2.waitKey()
    # my_cv_imwrite('D:\\study\\medical_picture\\三院\\三院\\code\\data\\'
    #               + os.path.basename(file_path).split('.')[0] + '-.png', img2)


#
# img = my_cv_imread('D:\\study\\medical_picture\\三院\\三院\\code\\test\\454.png')
# draw_frame(img, z_now=454, area_key=1000, z_toal=520)
# draw_frame('D:\\study\\medical_picture\\三院\\三院\\code\\data\\test\\c063.png', area_key=1000)
# dir = os.listdir(r'D:\study\medical_picture\三院\三院\code\data')
# # num_fal = []
# for temp in dir:
#     file = os.path.join(r'D:\study\medical_picture\三院\三院\code\data', temp)
#     # if os.path.isfile(file) and file.split('.')[0][-1] == '-':
#     #     os.remove(file)
#     if os.path.isfile(file) and file.split('.')[1] == 'png':
#         draw_frame(file, area_key=1000, z_toal=238)
# # print(num_fal)
# print(num_fal[np.argmin(num_fal)])
