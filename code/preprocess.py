import nibabel as nib
from nibabel import minc2
import numpy as np
import pdb
from matplotlib import pyplot as plt
import os
if __name__ == "__main__":
    dir_path = './data/huangcuiying'
    path_ct = os.path.join(dir_path,'CT_20180918_135647.nii.gz')
    nii_ct = nib.load(path_ct)
    data_ct = nii_ct.get_fdata()

    path_seg = os.path.join(dir_path,'Segmentation.seg.nii.gz')
    nii_seg = nib.load(path_seg)
    data_seg = nii_seg.get_fdata()

    print(data_ct.shape)
    print(data_seg.shape)

    ct_normal = data_ct[data_ct>-1000]
    ct_normal = ct_normal[ct_normal<1000]

    disease = data_ct[data_seg==1]

    ct_normal_no_disease = data_ct[data_seg==0]
    ct_normal_no_disease = ct_normal_no_disease[ct_normal_no_disease>-1000]
    ct_normal_no_disease = ct_normal_no_disease[ct_normal_no_disease<1000]

    ct_normal_part = data_ct[data_ct>-250]
    ct_normal_part = ct_normal_part[ct_normal_part<700]


    # fig, ax = plt.subplots()
    # ax.hist(x=ct_normal,bins=2001)
    # plt.savefig(os.path.join(dir_path,'total.png'))
    #
    # bins = int(np.max(disease)-np.min(disease)+1)
    # fig2, ax2 = plt.subplots()
    # ax2.hist(x=disease,bins=bins)
    # plt.savefig(os.path.join(dir_path,'disease.png'))
    #
    # bins3 = int(np.max(ct_normal_no_disease)-np.min(ct_normal_no_disease)+1)
    # fig3, ax3 = plt.subplots()
    # ax3.hist(x=ct_normal_no_disease,bins=bins3)
    # plt.savefig(os.path.join(dir_path,'no_disease.png'))

    # min_4 = int(min(np.min(ct_normal_part),np.min(disease)))
    # max_4 = int(max(np.max(ct_normal_part),np.max(disease)))
    fig4, ax4 = plt.subplots()
    # ax4.hist(x=ct_normal_part,bins=range(min_4,max_4+1),alpha=0.5,facecolor='blue')
    ax4.hist(x=disease,bins=range(int(np.min(disease)),int(np.max(disease))+1),alpha=0.5,facecolor='green')
    # ax4.hist(x=ct_normal_part,bins=range(min_4,max_4+1),alpha=0.5,facecolor='blue')
    plt.savefig(os.path.join(dir_path,'ct_part_disease.png'))
    plt.legend(loc='upper right')
    plt.show()

    # bins5 = range(int(np.min(ct_normal_part-1)),int(np.max(ct_normal_part+1)))
    # fig5, ax5 = plt.subplots()
    # ax5.hist(x=ct_normal_part,bins=bins5,alpha=0.5,facecolor='green')
    # plt.savefig(os.path.join(dir_path,'ct_part.png'))

    

