import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import math
import random
import csv
import cc3d


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


# 去除非人体的部分
def remove_nonhuman_part(img_data):
    img_sum = np.zeros((img_data.shape[0], img_data.shape[1]), np.uint8)
    for index in range(int(img_data.shape[2] * 0.2), int(img_data.shape[2] * 0.7)):
        img_sum = img_sum + img_data[:, :, index]
    ret, binary = cv2.threshold(img_sum, 1, 255, cv2.THRESH_BINARY)
    mask = np.zeros((img_data.shape[0], img_data.shape[1]), np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    area = stats[:, 4]
    area[0] = -1
    max_index = np.argmax(area)
    area[max_index] = -1
    max_index = np.argmax(area)
    mask[labels == max_index] = 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    for i in range(stats[max_index, 1] + stats[max_index, 3] - 5, mask.shape[0]):
        for j in range(stats[max_index, 0], stats[max_index, 0] + stats[max_index, 2]):
            mask[i, j] = 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(mask, contours, -1, 255, -1)
    mask = np.array(mask, dtype=np.bool_)
    for index in range(img_data.shape[2]):
        img_data[:, :, index][mask] = 0
    return img_data


# 低阈值的方法提取椎骨,仅提取椎骨
def find_max_region(img, center, area_key=100):
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
            dis.append((center[0] - centroids[i][0]) ** 2 + (center[1] - centroids[i][1]) ** 2)
        else:
            dis.append(1e9)
    minS = np.argmin(dis)
    output[labels == minS] = 255
    kernel = np.ones((5, 5), np.uint8)
    output_close = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel)
    contours, hierarch = cv2.findContours(output_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(output_close, contours, -1, color=255, thickness=-1)
    return output_close


# 去除脑中的组织
def remove_in_brain(img):
    # img = my_cv_imread(file_path)
    ret, binary = cv2.threshold(img, 75, 255, cv2.THRESH_BINARY)
    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    # 设置遮罩
    output = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    areas = stats[:, 4]
    areas[0] = 0
    index_brain = np.argmax(areas)
    output[labels == index_brain] = 255

    ret, binary = cv2.threshold(output, 75, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = stats[:, 4]
    areas[0] = 0
    areas[1] = 0
    index_brain_main = np.argmax(areas)
    output[:, :] = 0
    if areas[index_brain_main] == 0:
        return
    output[labels == index_brain_main] = 255
    return output


def detect_developer(img, area_key=800):
    ret, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    body_frame = get_frame_body(img)
    center = [(body_frame[0] + body_frame[1]) / 2, (body_frame[2] + body_frame[3]) / 2]
    body_dis = (body_frame[0] - body_frame[1]) ** 2 + (body_frame[2] - body_frame[3]) ** 2

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
    output_close = find_max_region(img, center, area_key)

    output[output_close == 255] = 255

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(output, connectivity=8)

    output2 = np.zeros((img.shape[0], img.shape[1]), np.uint8)

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
            if stats[i][4] >= 1000:
                dis_x.append(abs(centroids[i][0] - center[0]))
            else:
                dis_x.append(999)
        first_bone = []
        for i in range(1, num_labels):
            if dis_x[i] != 999:
                for j in range(i + 1, num_labels):
                    # 判断对称:x和y的坐标以及面积信息
                    # print(abs(dis_x[i] - dis_x[j]), abs(centroids[i][1] - centroids[j][1]), stats[i][4], stats[j][4],
                    #       abs(stats[i][4] - stats[j][4]))
                    # temp = np.zeros((img.shape[0], img.shape[1]), np.uint8)
                    # temp[labels == i] = 255
                    # temp[labels == j] = 255
                    # cv2.imshow('test', temp)
                    # cv2.waitKey()
                    if abs(dis_x[i] - dis_x[j]) <= 30 and abs(centroids[i][1] - centroids[j][1]) <= 30:
                        first_bone.append(i)
                        first_bone.append(j)
                        dis_x[j] = 999
                        break
        first_bone = np.array(first_bone)
        for i in range(1, num_labels):
            # 判断是否在人体轮廓外(患者的手部)和是否处在对称的组中
            if body_frame[0] < centroids[i][0] < body_frame[1] \
                    and body_frame[2] < centroids[i][1] < body_frame[3] and (i not in first_bone):
                check_area = stats[i][4] <= 20000
                # 计算相对距离并判断是否在(0.04, 0.35)之间 较远的话应该面积较小
                dis_temp = (centroids[i][0] - center[0]) ** 2 + (centroids[i][1] - center[1]) ** 2
                dis_relative = pow(dis_temp / body_dis, 0.5)
                check_dis_relative = 0.042 <= dis_relative < 0.24 or (
                        stats[i, 4] < 2000 and 0.24 <= dis_relative <= 0.35)
                # 计算x,y方向上的绝对距离和相对距离,判定y方向上在中心点以上 或 x方向上过度远离中心点
                check_dis_xandy = ((centroids[i][1] < center[1] + 10) or abs(
                    (centroids[i][0] - center[0]) / (body_frame[0] - body_frame[1])) >= 0.35)
                # 综合性地判断区域的距离
                check_dis = check_dis_relative and check_dis_xandy
                # if not check_dis:
                #     output2[labels == i] = 255
                #     continue
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
                if check_pos and check_dis and check_area:
                    # print(i, 'true:', dis_relative, stats[i][4], centroids[i][1], center[1] + 10,
                    #       round(abs((centroids[i][0] - center[0]) / (body_frame[0] - body_frame[1])), 2),
                    #       stats[i][2] ** 2 + stats[i][3] ** 2, stats[i][1] + stats[i][3],
                    #       centroids[i][0] - center[0], stats[i][2] * stats[i][3] / stats[i][4])
                    # out_temp = np.zeros((output.shape[0], output.shape[1]), np.uint8)
                    # out_temp[labels == i] = 255
                    # cv2.imshow('temp', out_temp)
                    # cv2.waitKey()
                    continue
                else:
                    # print(i, 'false:', dis_relative, stats[i][4], centroids[i][1], center[1] + 10,
                    #       round(abs((centroids[i][0] - center[0]) / (body_frame[0] - body_frame[1])), 2),
                    #       stats[i][2] ** 2 + stats[i][3] ** 2, stats[i][1] + stats[i][3],
                    #       centroids[i][0] - center[0], stats[i][2] * stats[i][3] / stats[i][4])
                    # out_temp = np.zeros((output.shape[0], output.shape[1]), np.uint8)
                    # out_temp[labels == i] = 255
                    # cv2.imshow('temp', out_temp)
                    # cv2.waitKey()
                    output2[labels == i] = 255
            else:
                output2[labels == i] = 255
    else:
        output2 = output
    contours, hierarch = cv2.findContours(output2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(output2, contours, -1, color=255, thickness=-1)
    return output2


# 三维张量上处理数据，并返回处理的一部分结果
def process_3D(nrrd_data):
    img_data_ori = np.clip(nrrd_data, 0, 255)
    img_data_ori = img_data_ori.astype(np.uint8)
    for i in range(img_data_ori.shape[2]):
        img_data_ori[:, :, i] = np.rot90(img_data_ori[:, :, i], k=-1)
    img_data_ori = remove_nonhuman_part(img_data_ori)
    temp = detect_developer(img_data_ori[:, :, 129])
    img_data_ori[:, :, 129][temp == 0] = 0

    # new_dir = os.path.join(r'D:\study\medical_picture\三院\三院\medical_picture\code',
    #                        os.path.basename(file).split('.')[0])
    # if not os.path.exists(new_dir):
    #     os.mkdir(new_dir)
    # for z in range(img_data_ori.shape[2]):
    #     file_path = new_dir + '/%03i' % z + '.png'
    #     img_temp_ori = img_data_ori[:, :, z]
    #     cv2.imencode('.png', img_temp_ori)[1].tofile(file_path)
    #     print(file_path)

    img_data = img_data_ori.copy()
    for z in range(img_data.shape[2]):
        ret, binary = cv2.threshold(img_data[:, :, z], 127, 255, cv2.THRESH_BINARY)
        img_data[:, :, z] = binary

    cc3d.dust(img_data, threshold=10000, in_place=True)
    labels_out, N = cc3d.connected_components(img_data, return_N=True)
    stats_label = cc3d.statistics(labels_out)
    label_area = stats_label['voxel_counts']
    label_area[0] = 0
    label_body_index = np.argmax(label_area)

    # img_temp = img_data[:, int(img_data.shape[1] / 2), :]
    # label_temp = labels_out[:, int(img_data.shape[1] / 2), :]
    # img_temp[label_temp == label_body_index] = 255
    # img_temp[label_temp != label_body_index] = 100
    # img_temp = np.rot90(img_temp)
    # cv2.imshow('temp', img_temp)
    # cv2.waitKey()

    # for index in range(20, 70):
    #     labels_out_temp = labels_out[:, :, index]
    #     img__temp = np.zeros((img_data.shape[0], img_data.shape[1], 3), np.uint8)
    #     for i in range(1, N + 1):
    #         img__temp[labels_out_temp == i] = (rd.randint(0, 255), rd.randint(0, 255), rd.randint(0, 255))
    #     cv2.imshow('temp', img__temp)
    #     cv2.waitKey()

    img_sum = np.zeros((img_data.shape[0], img_data.shape[1]), np.uint8)
    for index in range(int(img_data.shape[2] * 0.4), int(img_data.shape[2] * 0.6)):
        img_frame = img_data[:, :, index]
        ret, binary = cv2.threshold(img_frame, 127, 255, cv2.THRESH_BINARY)
        img_bool = np.array(binary, dtype=np.bool_)
        img_sum = img_sum + img_bool

    img_sum = np.array(img_sum, np.bool_)
    img_output = np.zeros((img_data.shape[0], img_data.shape[1]), np.uint8)
    img_output[img_sum] = 255
    # cv2.imshow('test', img_output)

    body_frame_xy = get_frame_body(img_output)
    center = [(body_frame_xy[0] + body_frame_xy[1]) / 2, (body_frame_xy[2] + body_frame_xy[3]) / 2]

    img_sum_vertical = np.zeros((img_data.shape[2], img_data.shape[0]), np.uint8)
    for i in range(int(center[0] - 20), int(center[0] + 20)):
        img_temp = img_data[:, i, :]
        img_temp = np.rot90(img_temp)
        _, binary_temp = cv2.threshold(img_temp, 127, 255, cv2.THRESH_BINARY)
        img_bool = np.array(binary_temp, dtype=np.bool_)
        img_sum_vertical = img_sum_vertical + img_bool
    img_sum_vertical = np.array(img_sum_vertical, dtype=np.bool_)
    img_output_vertical = np.zeros((img_sum_vertical.shape[0], img_sum_vertical.shape[1]), np.uint8)
    img_output_vertical[img_sum_vertical] = 255

    num_labels_pre, labels_pre, stats_pre, centroids_pre = cv2.connectedComponentsWithStats(img_output_vertical,
                                                                                            connectivity=8)
    areas_pre = stats_pre[:, 4]
    areas_pre[0] = 0
    index_body_pre = np.argmax(areas_pre)

    img_sum_vertical[:, :] = 0
    for i in range(int(center[0] - 32), int(center[0] - 20)):
        img_temp = img_data[:, i, :]
        img_temp = np.rot90(img_temp)
        _, binary_temp = cv2.threshold(img_temp, 127, 255, cv2.THRESH_BINARY)
        binary_temp[0:int(stats_pre[index_body_pre, 1] + stats_pre[index_body_pre, 3] * 0.3), :] = 0
        binary_temp[int(stats_pre[index_body_pre, 1] + stats_pre[index_body_pre, 3] * 0.8)
                    :binary_temp.shape[0], :] = 0
        img_bool = np.array(binary_temp, dtype=np.bool_)
        img_sum_vertical = img_sum_vertical + img_bool
    img_sum_vertical = np.array(img_sum_vertical, dtype=np.bool_)
    img_output_vertical[img_sum_vertical] = 255

    img_sum_vertical[:, :] = 0
    for i in range(int(center[0] + 20), int(center[0] + 32)):
        img_temp = img_data[:, i, :]
        img_temp = np.rot90(img_temp)
        _, binary_temp = cv2.threshold(img_temp, 127, 255, cv2.THRESH_BINARY)
        binary_temp[0:int(stats_pre[index_body_pre, 1] + stats_pre[index_body_pre, 3] * 0.3), :] = 0
        binary_temp[int(stats_pre[index_body_pre, 1] + stats_pre[index_body_pre, 3] * 0.8)
                    :binary_temp.shape[0], :] = 0
        img_bool = np.array(binary_temp, dtype=np.bool_)
        img_sum_vertical = img_sum_vertical + img_bool
    img_sum_vertical = np.array(img_sum_vertical, dtype=np.bool_)
    img_output_vertical[img_sum_vertical] = 255

    img_detect_brain = np.zeros((img_output_vertical.shape[0], img_output_vertical.shape[1]), np.uint8)
    img_detect_brain[labels_pre == index_body_pre] = 255
    _, img_detect_brain = cv2.threshold(img_detect_brain, 75, 255, cv2.THRESH_BINARY_INV)
    num_labels_brain, labels_brain, stats_brain, centroids_brain = \
        cv2.connectedComponentsWithStats(img_detect_brain, connectivity=8)
    areas_brain = stats_brain[:, 4]
    areas_brain[0] = 0
    areas_brain[1] = 0
    index_brain = np.argmax(areas_brain)
    brain_z = np.full((img_data.shape[2]), False, dtype=np.bool_)
    brain_z[img_data_ori.shape[2] - int(stats_brain[index_brain, 1] + stats_brain[index_brain, 3] + 2):
            img_data_ori.shape[2] - int(stats_brain[index_brain, 1] - 2)] = True

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_output_vertical, connectivity=8)
    areas = stats[:, 4]
    areas[0] = 0
    max_area_index = np.argmax(areas)
    img_output_vertical[labels == max_area_index] = 100
    centroids_y = centroids[:, 1]
    centroids_y[0] = 0
    maxy_index = 0
    while maxy_index == 0:
        if stats[np.argmax(centroids_y), 4] > 200:
            maxy_index = np.argmax(centroids_y)
        else:
            centroids_y[np.argmax(centroids_y)] = 0
    img_output_vertical[labels == maxy_index] = 100

    ribs_index = 0
    ribs_index_vertical = 0
    need_detect_z = np.full((img_data.shape[2]), False, dtype=np.bool_)
    for i in range(int(stats[max_area_index, 1] + stats[max_area_index, 3] * 0.3),
                   int(stats[max_area_index, 1] + stats[max_area_index, 3])):
        flag = 0
        for j in range(int(stats[max_area_index, 0]),
                       int(stats[max_area_index, 0] + stats[max_area_index, 2] * 0.5)):
            if img_output_vertical[i, j] == 255 and stats[labels[i, j], 4] > 500:
                index_area = labels[i, j]
                for k in range(int(center[0] - 30), int(center[0] + 30)):
                    for ribs_x in range(int(stats[index_area, 1]),
                                        int(stats[index_area, 1] + stats[index_area, 3])):
                        for rib_y in range(int(stats[index_area, 0]),
                                           int(stats[index_area, 0] + stats[index_area, 2])):
                            ribs_index = labels_out[rib_y, k, labels_out.shape[2] - ribs_x]
                            if ribs_index != 0:
                                break
                        if ribs_index != 0:
                            break
                    if ribs_index != 0:
                        break
                img_output_vertical[labels == index_area] = 50
                ribs_index_vertical = index_area
                flag = 1
                break
        if flag:
            break
    for i in range(int(stats[max_area_index, 1] + stats[max_area_index, 3]),
                   int(stats[max_area_index, 1] + stats[max_area_index, 3] * 0.3), -1):
        for j in range(int(stats[max_area_index, 0]),
                       int(stats[max_area_index, 0] + stats[max_area_index, 2] * 0.8)):
            if img_output_vertical[i, j] == 255 and labels[i, j] != 0:
                index_area = labels[i, j]
                # img_output_vertical[stats[index_area, 1]:stats[index_area, 1] + stats[index_area, 3],
                # stats[index_area, 0]] = 250
                # img_output_vertical[stats[index_area, 1]:stats[index_area, 1] + stats[index_area, 3],
                # stats[index_area, 0] + stats[index_area, 2]] = 250
                # img_output_vertical[stats[index_area, 1],
                # stats[index_area, 0]:stats[index_area, 0] + stats[index_area, 2]] = 250
                # img_output_vertical[stats[index_area, 1] + stats[index_area, 3],
                # stats[index_area, 0]:stats[index_area, 0] + stats[index_area, 2]] = 250
                # img_output_vertical[index_area == labels] = 250
                label_index = 0
                for k in range(int(center[0] - 30), int(center[0] + 30)):
                    if (labels_out[j, k, labels_out.shape[2] - i]) != 0:
                        label_index = (labels_out[j, k, labels_out.shape[2] - i])
                        break
                if label_index != label_body_index and label_index != ribs_index:
                    img_data[labels_out == label_index] = 0
                elif label_index == label_body_index and i > stats[ribs_index_vertical, 1] + stats[
                    ribs_index_vertical, 3]:
                    need_detect_z[img_data_ori.shape[2] - int(stats[index_area, 1] + stats[index_area, 3] + 5):
                                  img_data_ori.shape[2] - int(stats[index_area, 1] - 5)] = True
                img_output_vertical[labels == index_area] = 250
    need_detect_z[img_data_ori.shape[2] - int(stats[max_area_index, 1] + stats[max_area_index, 3] * 0.98):
                  img_data_ori.shape[2] - int(
                      stats[max_area_index, 1] + stats[max_area_index, 3] * 0.82)] = True
    return img_data_ori, img_data, need_detect_z, brain_z


# 逐帧处理
def detect_single(img_data_ori, img_data, need_detect_z, brain_z, z, area_key):
    img_temp_ori = img_data_ori[:, :, z]
    if need_detect_z[z]:
        temp = img_data[:, :, z].copy()
        img_data[:, :, z] = detect_developer(img_data_ori[:, :, z], area_key)
        img_data[:, :, z][temp == 0] = 0
    contours, _ = cv2.findContours(img_data[:, :, z], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_temp_mask = np.zeros((img_data.shape[0], img_data.shape[1]), np.uint8)
    cv2.drawContours(img_temp_mask, contours, -1, 255, -1)
    img_data_ori[:, :, z][img_temp_mask == 0] = 0
    if brain_z[z]:
        img_temp_ori[remove_in_brain(img_data_ori[:, :, z]) == 255] = 0
    return img_temp_ori


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


def draw_frame(img, z_now, area_key=100, z_total=238):
    z_rate = z_now / z_total

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
