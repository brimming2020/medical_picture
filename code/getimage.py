import time

import numpy as np
import nrrd
import os
import cv2
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from detect import get_frame_body


def delete_nonhuman_part(img_data):
    img_sum = np.zeros((img_data.shape[0], img_data.shape[1]), np.uint8)
    for index in range(int(img_data.shape[2] * 0.5), int(img_data.shape[2])):
        img_sum = img_sum + img_data[:, :, index]
    ret, binary = cv2.threshold(img_sum, 1, 255, cv2.THRESH_BINARY)
    mask = np.zeros((img_data.shape[0], img_data.shape[1]), np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    area = stats[:, 4]
    area[0] = -1
    max_index = np.argmax(area)
    mask[labels == max_index] = 255
    contours, hierarch = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(mask, contours, -1, color=255, thickness=-1)
    cv2.imshow('mask', mask)
    cv2.waitKey()
    mask = np.array(mask, dtype=np.bool_)
    for index in range(img_data.shape[2]):
        img_data[:, :, index][mask == 0] = 0
    return img_data


def test():
    print('test')
    time.sleep(2)


def temp():
    if run_state['s'] != 1:
        executor.submit(test())
        run_state['s'] = 1


executor = ProcessPoolExecutor(3)
run_state = {}
if __name__ == "__main__":
    # run_state['s'] = 0
    # temp()
    # temp()
    # nrrd图片读取
    dir = os.listdir(r'D:\study\medical_picture\三院\三院\medical_picture\code\nrrd')
    for file in dir:
        file = os.path.join(r'D:\study\medical_picture\三院\三院\medical_picture\code\nrrd', file)
        if os.path.isfile(file) and file.split('.')[1] == 'nrrd':
            if os.path.basename(file)[0] == 'C':
                continue
            nrrd_data, nrrd_options = nrrd.read(file)
            img_data = np.clip(nrrd_data, 0, 255)
            img_data = img_data.astype(np.uint8)
            for i in range(img_data.shape[2]):
                img_data[:, :, i] = np.rot90(img_data[:, :, i], k=-1)
            img_data = delete_nonhuman_part(img_data)

            img_sum = np.zeros((img_data.shape[0], img_data.shape[1]), np.uint8)
            for index in range(int(img_data.shape[2] * 0.42), int(img_data.shape[2] * 0.63)):
                img_frame = img_data[:, :, index]
                ret, binary = cv2.threshold(img_frame, 127, 255, cv2.THRESH_BINARY)
                img_bool = np.array(binary, dtype=np.bool_)
                img_sum = img_sum + img_bool

            img_sum = np.array(img_sum, np.bool_)
            img_output = np.zeros((img_data.shape[0], img_data.shape[1]), np.uint8)
            img_output[img_sum] = 255
            # cv2.imshow('test', img_output)

            body_frame = get_frame_body(img_output)
            body_dis = (body_frame[0] - body_frame[1]) ** 2 + (body_frame[2] - body_frame[3]) ** 2
            center = [(body_frame[0] + body_frame[1]) / 2, (body_frame[2] + body_frame[3]) / 2]

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

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_output_vertical, connectivity=8)
            # 过滤离散的小白点
            for i in range(1, num_labels):
                if stats[i][4] < 100:
                    img_output_vertical[i == labels] = 0
            # cv2.imshow(os.path.basename(file), img_output_vertical)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_output_vertical, connectivity=8)
            areas = stats[:, 4]
            areas[0] = -1
            max_area_index = np.argmax(areas)
            centroids_y = centroids[:, 1]
            centroids_y[0] = 0
            maxy_index = np.argmax(centroids_y)
            img_output_vertical[labels == maxy_index] = 50
            img_output_vertical[labels == max_area_index] = 50
            body_frame = get_frame_body(img_output_vertical)
            img_output_vertical[body_frame[2], body_frame[0]:body_frame[1]] = 250
            img_output_vertical[body_frame[3], body_frame[0]:body_frame[1]] = 250
            img_output_vertical[body_frame[2]:body_frame[3], body_frame[0]] = 250
            img_output_vertical[body_frame[2]:body_frame[3], body_frame[1]] = 250
            for i in range(int(body_frame[2] + (body_frame[3] - body_frame[2]) * 0.3), body_frame[3]):
                flag = 0
                for j in range(int(body_frame[0] + (body_frame[1] - body_frame[0]) * 0.1),
                               int(body_frame[0] + (body_frame[1] - body_frame[0]) * 0.5)):
                    if img_output_vertical[i, j] == 255 and labels[i, j] != max_area_index \
                            and stats[labels[i, j], 4] > 500:
                        index_area = labels[i, j]
                        img_output_vertical[labels == index_area] = 50
                        flag = 1
                        break
                if flag:
                    break
            for i in range(body_frame[3], int(body_frame[2] + (body_frame[3] - body_frame[2]) * 0.4), -1):
                for j in range(int(body_frame[0] + (body_frame[1] - body_frame[0]) * 0.1),
                               int(body_frame[0] + (body_frame[1] - body_frame[0]) * 0.8)):
                    if img_output_vertical[i, j] == 255 and labels[i, j] != 0:
                        index_area = labels[i, j]
                        print(stats[index_area])
                        print(img_output_vertical.shape)
                        # img_output_vertical[i, :] = 240
                        # img_output_vertical[:, j] = 240
                        img_output_vertical[stats[index_area, 1]:stats[index_area, 1] + stats[index_area, 3],
                        stats[index_area, 0]] = 255
                        img_output_vertical[stats[index_area, 1]:stats[index_area, 1] + stats[index_area, 3],
                        stats[index_area, 0] + stats[index_area, 2]] = 250
                        img_output_vertical[stats[index_area, 1],
                        stats[index_area, 0]:stats[index_area, 0] + stats[index_area, 2]] = 250
                        img_output_vertical[stats[index_area, 1] + stats[index_area, 3],
                        stats[index_area, 0]:stats[index_area, 0] + stats[index_area, 2]] = 250
                        img_output_vertical[index_area == labels] = 250
            cv2.imshow(os.path.basename(file), cv2.resize(img_output_vertical, (
                int(img_output_vertical.shape[1] * nrrd_options['space directions'][0][0]),
                int(img_output_vertical.shape[0] * nrrd_options['space directions'][2][2]))))

            # 开运算把相连的邻域分开
            kernel = np.ones((2, 2), np.uint8)
            img_open = cv2.morphologyEx(img_output, cv2.MORPH_OPEN, kernel)
            kernel = np.ones((5, 5), np.uint8)
            img_close = cv2.dilate(img_open, kernel, iterations=1)
            # 连通域分析
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_close, connectivity=8)
            output = np.zeros((img_output.shape[0], img_output.shape[1]), np.uint8)
            # 过滤离散的小白点
            for i in range(1, num_labels):
                if stats[i][4] > 100:
                    output[i == labels] = 255
            # cv2.imshow('output', output)
            # cv2.waitKey()
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(output, connectivity=8)
            num = 0
            for i in range(1, num_labels):
                dis_temp = (centroids[i][0] - center[0]) ** 2 + (centroids[i][1] - center[1]) ** 2
                dis_relative = pow(dis_temp / body_dis, 0.5)
                if dis_relative < 0.45:
                    num += 1
    cv2.waitKey()
    # print("begin write")
    # for i in range(0, len1):
    #     num = str(i)
    #     nrrd_get = nrrd_data[:, :, i]
    #     if i < 100:
    #         num = '0' + num
    #         if i < 10:
    #             num = '0' + num
    #     for ii in range(0, 512):
    #         for j in range(0, 512):
    #             if opening[ii, j] == 255:
    #                 nrrd_get[ii, j] = 0
    #     nrrd_img = Image.fromarray(nrrd_get)
    #     nrrd_img.show()
    #     nrrd_img.save('111.png', 'PNG')
    #     im = Image.open('222.png')
    #     im.show()
    #
    #     cv2.imwrite('temp.png', nrrd_get)
    #     temp = cv2.imread('temp.png', flags=0)
    #     print(nrrd_get)
    #     print(temp)
    #     print(np.array(nrrd_img))
    #     cv2.imshow('222', temp)
    #     cv2.waitKey()

    # nrrd_image = Image.fromarray(nrrd_get)
    # nrrd_image.save(nrrd_image/255.0,'./code/lab/M0'+str(i)+'.png')

    # nrrd_data[:,:,29] 表示截取第30张切片
    # nrrd_image.show() # 显示这图片
    print('ss')
