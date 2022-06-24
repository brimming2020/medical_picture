import nibabel as nib
from nibabel import minc2
import numpy as np
import pdb
from matplotlib import pyplot as plt
import cv2
import os
import scipy
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from generate_images import generateImages
from mpl_toolkits.mplot3d import Axes3D


def f(M, abc, i, j, k):
    """ Return X, Y, Z coordinates for i, j, k """
    return M.dot([i, j, k]) + abc


def resample(image, spacing, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    # spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing


def plot_3d(image, threshold=-300, face_color=[0.45, 0.45, 0.75]):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)

    verts, faces, _, _ = measure.marching_cubes_lewiner(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    # face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])


def extract_bone(file_path, thres=400):
    img_nib = nib.load(file_path)
    affine = img_nib.affine
    img_header = img_nib.header
    img_data = img_nib.get_data()
    # img_data = np.rot90(img_data,k=-1,axes=(0,2))

    img_data[img_data < thres] = 0
    img_data[img_data >= thres] = 1

    return img_data, affine, img_header


def main():
    file_path = "./data/normal/normal-pa9-lijinhua/3 WB Standard.nii.gz"
    output_path = "./data/normal/chengtie/plt"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    img_nib = nib.load(file_path)
    img_data = img_nib.get_data()
    affine = img_nib.affine
    img_data = np.rot90(img_data, k=-1, axes=(0, 2))
    img_copy = img_data.copy()

    # img_data = np.clip(img_data,0,255)
    # img_data = img_data.astype(np.uint8)
    #
    # for i in range(img_data.shape[2]):
    #     slice_data = img_data[:, :, i]
    #     _, thre = cv2.threshold(slice_data,127,255,cv2.THRESH_BINARY)#cv2.THRESH_TOZERO)
    #     img_data[:,:,i] = thre

    # for i in range(300,400,10):
    #     print(i)
    #     img_copy[img_data < i] = 0
    #     img_copy[img_data >= i] = 255
    #
    #     new_image = nib.Nifti1Image(img_copy, affine)
    #     nib.save(new_image, 'new_image-{}.nii.gz'.format(i))
    #
    #     img_header = img_nib.header
    #     img_pixdim = img_header['pixdim']
    #     spacing = img_pixdim[1:4]
    #
    #     print("resample...", end='', flush=True)
    #     img_data_resampled, spacing = resample(img_copy,spacing)
    #     print("done", flush=True)
    #     print('ploting...', end='', flush=True)
    #     plot_3d(img_data_resampled,threshold=120)
    #     print("done", flush=True)
    #     print("saving...", end='', flush=True)
    #     # plt.show()
    #     plt.savefig(os.path.join(output_path,'debug-{}.png'.format(i)))
    #     plt.clf()

    img_copy[img_data < 350] = 0
    img_copy[img_data >= 350] = 255


if __name__ == "__main__":
    main()
