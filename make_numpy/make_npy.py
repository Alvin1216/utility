#%%
import labelme
from labelme import utils
import os, cv2, random, re , json
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

#%%
IMAGE_HEIGH = 512
IMAGE_WIDTH = 512

#%%
def generate_label_png(json_file_path):
    print('generate_label_png'+json_file_path)
    data = json.load(open(json_file_path))
    #imageData = data['imageData']
    #img = utils.img_b64_to_arr(imageData)
    label_name_to_value = {'_background_': 0}
    for shape in data['shapes']:
        label_name = shape['label']
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value
            
    label_values, label_names = [], []
    for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
        label_values.append(lv)
        label_names.append(ln)
    print(label_names)
    assert label_values == list(range(len(label_values)))
    label_only = utils.shapes_to_label((512,512), data['shapes'], label_name_to_value).astype('uint8')
    #有多種標籤的話會生成不同顏色
    #就像最大連通區域
    #屬於第一種標籤的部分 塗上就會標為1
    #屬於第二種標籤的部分 塗上就會標為2
    #背景就會標成 0 
    #但是我現在想把它變成同一張 所以大於等於1的部分都會標成1
    label_only[label_only>=1]=1
    filename = data['imagePath']
    return label_only,filename

def save_to_npy(base_path,filename,nparray):
    #filename = filename + '.npz'
    savefile_path = os.path.join(base_path,filename)
    if not os.path.isdir(base_path):
        os.makedirs(base_path)
    #np.savez_compressed(savefile_path,nparray) 
    np.save(savefile_path,nparray) 
    return savefile_path

def filename_cheaker_png(filename):
    pattern = re.compile(r'^.*?.png$')
    match = pattern.match(filename)
    if match:
        return True
    else:
        return False

#image read from cv.imread is nparry and its dtype is unit8
#so the making black label need to be same dtype
def load_one_person_image(image_folder_path):
    original_all_image = []
    original_all_label = []
    image_has_segmentation = []
    #os.path.join()
    image_file_list = listdir(image_folder_path)
    image_file_list = [ name for name in image_file_list if filename_cheaker_png(name) == True ]
    image_file_list = sorted(image_file_list)
    print(image_file_list)

    for image_file_name in image_file_list:
        #read file that is .png
        if(filename_cheaker_png(image_file_name) == True):
            image_path = os.path.join(image_folder_path , image_file_name)
            image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
            
            #check resolution is 512x512
            if (image.shape != (512,512)):
                image = cv2.resize(image,(512,512))
                #print(image_path,' is not 512x512')
                #continue
            
            label_file_name = image_file_name.split('.')[0] + '.json'
            label_path = os.path.join(image_folder_path , label_file_name)
            #if json file is exist means that we have this slice's segmentation
            #otherwise we make a blank label(image is all black) for this slice
            if(os.path.isfile(label_path)):
                label,filename = generate_label_png(label_path)
                image_has_segmentation.append(1)
            else:
                label = np.zeros((IMAGE_HEIGH,IMAGE_WIDTH)).astype('uint8')
                image_has_segmentation.append(0)
            
            if label.shape != (512,512):
                print('no 512!')
                label = cv2.resize(label,(512,512))
            
            print('image size ', image.shape)
            print('label size ', label.shape)
            original_all_image.append(image)
            original_all_label.append(label)
    return original_all_image,original_all_label,image_has_segmentation

def save_one_person_npy(image_folder_path,patient_number):
    base_path = os.path.join('/Users/alvinhuang/Desktop/new_ich_npy/171207',patient_number)
    original_all_image,original_all_label,image_has_segmentation = load_one_person_image(image_folder_path)
    filename = patient_number + '_image'
    filepath = save_to_npy(base_path,filename,np.asarray(original_all_image))
    print('image done! npz in '+ filepath)
    
    filename = patient_number + '_label'
    filepath = save_to_npy(base_path,filename,np.asarray(original_all_label))
    print('label done! npz in '+ filepath)
    
    filename = patient_number + '_segment_flag'
    filepath = save_to_npy(base_path,filename,np.asarray(image_has_segmentation))
    print('flag done! npz in '+ filepath)

#%%
base_path = "/Users/alvinhuang/Desktop/ICH171-207"
for index,folder_name in zip(range(0,len(listdir(base_path))),sorted(listdir(base_path))):
    print(index,folder_name)
    if folder_name != '.DS_Store':
        image_folder_path = os.path.join(base_path,folder_name)
        print(image_folder_path)
        save_one_person_npy(image_folder_path,folder_name)
        # image_folder_path = os.path.join(base_path,folder_name)
        # print(image_folder_path)
        # save_one_person_npy(image_folder_path,folder_name)

#generate_label_png('/Users/alvinhuang/Desktop/ich0001_2/ICH0011/ICH0011VS17.json')

# %%
