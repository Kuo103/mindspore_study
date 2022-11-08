import numpy as np
import os, random, glob
from skimage import transform
from PIL import Image
import matplotlib.pyplot as plt
import nibabel as nib
import SimpleITK as sitk
import skimage.io as io
import mindspore as ms
import mindspore.dataset as ds


img_path='./model2_img_train/case_00000_1.nii.gz'
label_path='./model2_tumor_train/case_00000_1.nii.gz'
image=nib.load(img_path)
image=image.get_fdata()
label=nib.load(label_path)
label=label.get_fdata()
for i in range(25):
    plt.subplot(121)
    plt.imshow(image[i,:,:])
    plt.subplot(122)
    plt.imshow(label[i,:,:])
    plt.show()