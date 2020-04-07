#%%
import pydicom
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# %%
def transform_to_hu(medical_image, image):
    intercept = medical_image.RescaleIntercept
    slope = medical_image.RescaleSlope
    hu_image = image * slope + intercept

    return hu_image

# %%
def window_image(image, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    
    return window_image
# %%
file_path = "/Users/alvinhuang/Desktop/ICH0001-80_original_file/ICH0001-21/ICH0016/ICH0016VS04.dcm"
medical_image = pydicom.read_file(file_path)

# %%
image = medical_image.pixel_array
print(image.min())
print(image.max())
# %%
hu_image = transform_to_hu(medical_image,image)
brain_image = window_image(hu_image, 50, 100)
bone_image = window_image(hu_image, 300, 1500)
abol_image = window_image(hu_image, 40, 350)
ori_image = window_image(hu_image, 35, 90)

# %%
hu_image = transform_to_hu(medical_image,image)
brain_image = window_image(hu_image, 40, 80)
blood_image = window_image(hu_image, 80, 200)
soft_image = window_image(hu_image, 40, 380)
ori_image = window_image(hu_image, 35, 90)
# %%
plt.imshow(brain_image,cmap='gray')
plt.colorbar()

# %%
plt.imshow(bone_image,cmap='gray')
plt.colorbar()
# %%
plt.imshow(soft_image,cmap='gray')
plt.colorbar()

# %%
plt.imshow(ori_image,cmap='gray')
plt.colorbar()

# %%
import scipy.misc
out_path = 'output.jpg'
scipy.misc.imsave(out_path,bone_image)

# %%
