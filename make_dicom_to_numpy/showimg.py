#%%
import matplotlib.pyplot as plt
import numpy as np

image = np.load('/Users/alvinhuang/Desktop/new_ich_npy/6180/ICH0061/ICH0061_image.npy')
label = np.load('/Users/alvinhuang/Desktop/new_ich_npy/6180/ICH0061/ICH0061_label.npy')

index = 10
plt.imshow(image[index],cmap='gray')
plt.imshow(label[index],cmap='jet',alpha = 0.3)
# %%
def window_image(image, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    
    return window_image

# %%
wm = window_image(image[0],80,200)

# %%
plt.imshow(wm.astype('float64'),cmap='gray')
plt.colorbar()

# %%
