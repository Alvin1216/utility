import matplotlib.pyplot as plt
import numpy as np
import os, cv2, random

def save_to_npy(base_path,filename,nparray):
    #filename = filename + '.npz'
    savefile_path = os.path.join(base_path,filename)
    if not os.path.isdir(base_path):
        os.makedirs(base_path)
    #np.savez_compressed(savefile_path,nparray) 
    np.save(savefile_path,nparray) 
    return savefile_path

base_path = '/Users/alvinhuang/Desktop/new_ich_npy/6180'
all_image = []
all_label = []
all_flag = []
oneman_slices = [] #一個人有幾張
array_location = [] #在大陣列中 結束的位置 （從0開始）
folder_list = os.listdir(base_path)
folder_list = [x for x in folder_list if x != '.DS_Store']
folder_list = sorted(folder_list)
print(folder_list)

for folder_name,index in zip(folder_list,range(0,len(folder_list))):
    if folder_name == 'T66' or folder_name == '.DS_Store':
        print('66 bye/DS_Store bye!!')
        continue
    image_path = os.path.join(os.path.join(base_path,folder_name),folder_name+'_image.npy')
    label_path = os.path.join(os.path.join(base_path,folder_name),folder_name+'_label.npy')
    flag_path = os.path.join(os.path.join(base_path,folder_name),folder_name+'_segment_flag.npy')
    
    image = np.load(image_path,allow_pickle=True)
    print('image dtype',str(image.dtype))
    print(image_path,' has loaded.','shape: ',str(image.shape))
    if index == 0:
        all_image = image
    else:
        all_image = np.concatenate((all_image,image),axis=0)
    label = np.load(label_path,allow_pickle=True)
    print('label dtype',str(label.dtype))
    print(label_path,'has loaded','shape: ',str(label.shape))
    if index == 0:
        all_label = label
    else:
        all_label = np.concatenate((all_label,label),axis=0)
    flag = np.load(flag_path,allow_pickle=True)
    print('flag dtype',str(flag.dtype))
    print(flag_path,'has loaded','shape: ',str(flag.shape))
    if index == 0:
        all_flag = flag
    else:
        all_flag = np.concatenate((all_flag,flag),axis=0)
    print('all image shape: ',str(all_image.shape),'type: '+str(type(all_image)))
    print('all label shape: ',str(all_label.shape),'type: '+str(type(all_label)))
    print('all flag shape: ',str(all_flag.shape),'type: '+str(type(all_flag)))
    print('all image dtype: ',str(all_image.dtype))
    print('all label dtype: ',str(all_label.dtype))
    print('all flag dtype: ',str(all_flag.dtype))
    oneman_slices.append(image.shape[0])
    array_location.append(all_image.shape[0]-1)
    print('oneman_slices: ',str(image.shape[0]),'array_location: '+str(all_image.shape[0]-1))


folder_name = '6180'
base_path = '/Users/alvinhuang/Desktop/new_ich_npy/all_in_one/6180'
if not os.path.exists(base_path):
     os.mkdir(base_path)
save_to_npy(base_path,'image_'+folder_name,all_image)
save_to_npy(base_path,'label_'+folder_name,all_label)
save_to_npy(base_path,'flag_'+folder_name,all_flag)
save_to_npy(base_path,'one_man_slice_numbers_'+folder_name,oneman_slices)
save_to_npy(base_path,'one_man_slice_array_end_location_'+folder_name,array_location)