import numpy as np
import nrrd
import os
import cv2
import cc3d
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
    # nrrd图片读取
    dir = os.listdir(r'D:\study\medical_picture\三院\三院\medical_picture\code\nrrd')
    for file in dir:
        file = os.path.join(r'D:\study\medical_picture\三院\三院\medical_picture\code\nrrd', file)
        if os.path.isfile(file) and file.split('.')[1] == 'nrrd':
            nrrd_data, nrrd_options = nrrd.read(file)
            img_data = np.clip(nrrd_data, 0, 255)
            img_data = img_data.astype(np.uint8)
            print(img_data)
            for i in range(img_data.shape[2]):
                img_data[:, :, i] = np.rot90(img_data[:, :, i], k=-1)
            img_data = delete_nonhuman_part(img_data)
            # for i in range(img_data.shape[0]):
            #     for j in range(img_data.shape[1]):
            #         for k in range(img_data.shape[2]):
            #             if img_data[i, j, k] > 127:
            #                 img_data[i, j, k] = 255
            #             else:
            #                 img_data[i, j, k] = 0
            labels_out = measure.label(img_data, background=0)
            properties = measure.regionprops(labels_out)
            print(len(properties))
            for z in range(img_data.shape[2]):
                ret, binary = cv2.threshold(img_data[:, :, z], 127, 255, cv2.THRESH_BINARY)
                img_data[:, :, z] = binary

            labels_out, N = cc3d.connected_components(img_data, return_N=True)
            print(labels_out, N)
            img_data *= (labels_out > 0)

            body_frame = get_frame_body(img_output)
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
            cv2.imshow('temp', cv2.resize(img_output_vertical,
                                          (int(img_output_vertical.shape[1] * nrrd_options['space directions'][0, 0]),
                                           int(img_output_vertical.shape[0] * nrrd_options['space directions'][2, 2]))))
            cv2.waitKey()
