import numpy as np
import nrrd
import os
import cv2
from detect import get_frame_body


def delete_nonhuman_part(img_data):
    img_sum = np.zeros((img_data.shape[0], img_data.shape[1]), np.uint8)
    for index in range(int(img_data.shape[2] * 0.3), int(img_data.shape[2] * 0.6)):
        img_sum = img_sum + img_data[:, :, index]
    ret, binary = cv2.threshold(img_sum, 1, 255, cv2.THRESH_BINARY)
    mask = np.zeros((img_data.shape[0], img_data.shape[1]), np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    area = stats[:, 4]
    area[0] = -1
    max_index = np.argmax(area)
    mask[labels == max_index] = 255
    mask = np.array(mask, dtype=np.bool_)
    cv2.waitKey()
    for index in range(img_data.shape[2]):
        img_data[:, :, index][mask == 0] = 0
    return img_data


if __name__ == "__main__":
    # run_state['s'] = 0
    # temp()
    # temp()
    # nrrd图片读取
    dir = os.listdir(r'D:\study\medical_picture\三院\三院\medical_picture\code\nrrd')
    for file in dir:
        file = os.path.join(r'D:\study\medical_picture\三院\三院\medical_picture\code\nrrd', file)
        if os.path.isfile(file) and file.split('.')[1] == 'nrrd':
            nrrd_data, nrrd_options = nrrd.read(file)
            print(nrrd_options)
            img_data = np.clip(nrrd_data, 0, 255)
            img_data = img_data.astype(np.uint8)
            for i in range(img_data.shape[2]):
                img_data[:, :, i] = np.rot90(img_data[:, :, i], k=-1)
            img_data = delete_nonhuman_part(img_data)
            # if not os.path.exists(os.path.basename(file).split('.')[0]+'/'):
            #     os.mkdir(os.path.basename(file).split('.')[0])
            # for i in range(img_data.shape[2]):
            #     cv2.imwrite(os.path.basename(file).split('.')[0]+'/%03d.png' % i, img_data[:, :, i])

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

            # img_output_vertical = cv2.resize(img_output_vertical, (
            #     int(img_output_vertical.shape[1] * nrrd_options['space directions'][0][0]/2),
            #     int(img_output_vertical.shape[0] * nrrd_options['space directions'][2][2]/4)))
            cv2.imshow("temp_pre", img_output_vertical)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_output_vertical, connectivity=4)
            areas = stats[:, 4]
            areas[0] = -1
            max_area_index = np.argmax(areas)
            centroids_y = centroids[:, 1]
            centroids_y[0] = 0
            maxy_index = np.argmax(centroids_y)
            img_output_vertical[labels == maxy_index] = 50
            cv2.imshow("temp", img_output_vertical)
            cv2.waitKey()
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
            cv2.imshow(os.path.basename(file), img_output_vertical)
    cv2.waitKey()
