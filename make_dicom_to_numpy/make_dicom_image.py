import pydicom
import numpy as np
import cv2
import os,re
from os import listdir
import matplotlib.pyplot as plt

def transform_to_hu(original_image,intercept,slope):
    hu_image = original_image * slope + intercept
    return hu_image

def get_intercept_slope(medical_image):
    intercept = medical_image.RescaleIntercept
    slope = medical_image.RescaleSlope
    return (intercept,slope)

def get_original_dicom_array(medical_image):
    image = medical_image.pixel_array
    return image

def filename_cheaker_dcm(filename):
    pattern = re.compile(r'^.*?.dcm$')
    match = pattern.match(filename)
    if match:
        return True
    else:
        return False

def window_image(image, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    
    return window_image

def save_to_npy(base_path,filename,nparray):
    #filename = filename + '.npz'
    savefile_path = os.path.join(base_path,filename)
    if not os.path.isdir(base_path):
        os.makedirs(base_path)
    #np.savez_compressed(savefile_path,nparray) 
    np.save(savefile_path,nparray) 
    return savefile_path

def save_one_person_npy(patient_number,image):
    base_path = os.path.join('/Users/alvinhuang/Desktop/ich_dicom_numpy',patient_number)
    filename = patient_number + '_dicom_image'
    filepath = save_to_npy(base_path,filename,np.asarray(image))

base_path = "/Users/alvinhuang/Desktop/T01-T21_original_data"
for index,folder_name in zip(range(0,len(listdir(base_path))),sorted(listdir(base_path))):
    image_folder_path = os.path.join(base_path,folder_name)
    if folder_name != '.DS_Store':
        dicom_hu_array = []
        #try:
        for index,image_filename in zip(range(0,len(listdir(image_folder_path))),sorted(listdir(image_folder_path))):
            if filename_cheaker_dcm(image_filename):
                png_filename = image_filename.split('.')[0]
                png_filename = png_filename.split('-')[0] 
                number = png_filename.split('-')[0][-4:]
                png_filename = number + '-' + png_filename.split('-')[0]+'.png'
                print(png_filename)
                #print(os.path.exists(png_filename))
                if os.path.exists(os.path.join(image_folder_path,png_filename)):
                    print(image_filename)
                    image_path = os.path.join(image_folder_path,image_filename)
                    medical_image = pydicom.read_file(image_path)
                    image = get_original_dicom_array(medical_image)
                    intercept_slope = get_intercept_slope(medical_image)
                    print(intercept_slope)

                    if image.shape != (512,512):
                        import scipy.ndimage
                        xscale = 512/image.shape[0]
                        yscale = 512/image.shape[1]
                        image = scipy.ndimage.interpolation.zoom(image, [xscale, yscale])
                        print("image had been reshape! size: ",image.shape)
                        
                    hu_image = transform_to_hu(image,intercept_slope[0],intercept_slope[1])
                    print(hu_image.shape)
                    dicom_hu_array.append(hu_image)
        dicom_hu_array = np.asarray(dicom_hu_array).astype('float16')
        print(dicom_hu_array.shape)
        save_one_person_npy(folder_name,dicom_hu_array)
        print('dicom_hu_array',dicom_hu_array.dtype)
        #except:
        #    print(folder_name + 'wrong!')