import cc3d
import numpy as np
import pdb
from bone_extraction import extract_bone,resample,plot_3d
from collections import Counter
from matplotlib import pyplot as plt
import os
import nibabel as nib

def connect3d(image_data, most_common=30):
    connectivity = 6
    labels_out,N = cc3d.connected_components(image_data, connectivity=connectivity,return_N=True)
    label_common = dict(Counter(labels_out.flatten()).most_common(most_common))

    return labels_out, label_common

def main():
    file_path = "./data/normal/linguozhong/2 CT Atten Cor Head In.nii.gz"
    output_path = "./data/normal/linguozhong/plt"

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    bone_data, affine, image_header = extract_bone(file_path)

    connectivity = 6
    labels_out,N = cc3d.connected_components(bone_data, connectivity=connectivity,return_N=True)
    label_count = dict(Counter(labels_out.flatten()).most_common(30))
    # pdb.set_trace()

    img_pixdim = image_header['pixdim']
    spacing = img_pixdim[1:4]

    # for i,(key, value) in enumerate(label_count.items()):
    #     if key == 0 :
    #         continue
    #     extracted_image = labels_out * (labels_out == key)
    #     image_temp = bone_data*(labels_out == key)
    #     # pdb.set_trace()
    #     print("resample...", end='', flush=True)
    #     img_data_resampled, spacing = resample(image_temp,spacing)
    #     print("done", flush=True)
    #     print('ploting...', end='', flush=True)
    #     plot_3d(img_data_resampled,threshold=120)
    #     print("done", flush=True)
    #     print("saving...", end='', flush=True)
    #     plt.savefig(os.path.join(output_path,'debug-i-6-{}.png'.format(key)))
    #     print("done", flush=True)
    #     print(i, key, value)

    img_nib = nib.load(file_path)
    img_data = img_nib.get_data()
    img_data = np.rot90(img_data,k=-1,axes=(0,2))
    # fig4, ax4 = plt.subplots()
    for i,(key, value) in enumerate(label_count.items()):
        if key == 0:
            continue
        image_temp = img_data[labels_out==key]
        # pdb.set_trace()
        print('ploting...', end='', flush=True)
        plt.hist(x=image_temp,bins=10,alpha=0.5,facecolor='green')
        print("done", flush=True)
        print("saving...", end='', flush=True)
        plt.savefig(os.path.join(output_path,'debug-i-6-{}-2.png'.format(key)))
        print("done", flush=True)
        plt.clf()
        print(i, key, value)
        # break
        # break
    # for segid in range(1, N+1):
    #     extracted_image = labels_out * (labels_out == segid)
    #     sum = np.sum(extracted_image) / segid
    #     print("Id:{},nums:{}".format(segid,sum))
        # pdb.set_trace()

if __name__ == '__main__':
    main()

# labels_in = np.ones((512, 512, 512), dtype=np.int32)
# labels_out = cc3d.connected_components(labels_in) # 26-connected

# connectivity = 6 # only 4,8 (2D) and 26, 18, and 6 (3D) are allowed
# labels_out = cc3d.connected_components(labels_in, connectivity=connectivity)

# You can extract the number of labels (which is also the maximum
# label value) like so:
# labels_out, N = cc3d.connected_components(labels_in, return_N=True) # free
# -- OR --
# labels_out = cc3d.connected_components(labels_in)
# N = np.max(labels_out) # costs a full read

# You can extract individual components using numpy operators
# This approach is slow, but makes a mutable copy.
# for segid in range(1, N+1):
#     extracted_image = labels_out * (labels_out == segid)
#     pdb.set_trace()
    # process(extracted_image) # stand in for whatever you'd like to do

# If a read-only image is ok, this approach is MUCH faster
# if the image has many contiguous regions. A random image
# can be slower. binary=True yields binary images instead
# of numbered images.
# for label, image in cc3d.each(labels_out, binary=False, in_place=True):
#     process(image) # stand in for whatever you'd like to do

# Image statistics like voxel counts, bounding boxes, and centroids.
# stats = cc3d.statistics(labels_out)

# We also include a region adjacency graph function
# that returns a set of undirected edges.
# edges = cc3d.region_graph(labels_out, connectivity=connectivity)

# You can also generate a voxel connectivty graph that encodes
# which directions are passable from a given voxel as a bitfield.
# This could also be seen as a method of eroding voxels fractionally
# based on their label adjacencies.
# See help(cc3d.voxel_connectivity_graph) for details.
# graph = cc3d.voxel_connectivity_graph(labels, connectivity=connectivity)

