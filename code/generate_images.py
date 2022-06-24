import pdb

import nibabel as nib
import sys
import os
import cv2
import numpy as np
import warnings
import nrrd

warnings.filterwarnings("ignore")


def generateImages(input_nii_file_path, output_images_path):
    if not os.path.exists(output_images_path):
        os.makedirs(output_images_path)

    img_data = nib.load(input_nii_file_path).get_data()
    # img_data2 = np.clip(img_data,0,255)

    img_data = np.rot90(img_data, k=-1, axes=(0, 1))
    for i in range(img_data.shape[2]):
        curr_image = img_data[:, :, i]
        # pdb.set_trace()
        # curr_image = np.transpose(
        #     curr_image[::-1, ...][:, ::-1], axes=(1, 0))[::-1, ...]
        cv2.imwrite(output_images_path + '/' + 'IM%03d.png' % i, curr_image)
        # cv2.imwrite(output_images_path + '/' + prefix + 'IM%03d_skeleton_0.png' % i, curr_image)


if __name__ == '__main__':
    input_nrrd_file_path = "./Z/data4/CT_20200422_122905.nrrd"
    filename = os.path.basename(input_nrrd_file_path).split('.')[0]
    input_nii_file_path = os.path.dirname(input_nrrd_file_path)+'/'+filename+'.nii'

    data, options = nrrd.read(input_nrrd_file_path)
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, input_nii_file_path)

    output_images_path = os.path.dirname(input_nrrd_file_path)+'/'+filename
    generateImages(input_nii_file_path, output_images_path)
