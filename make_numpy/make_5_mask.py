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
    #label_only[label_only>=1]=1
    if label_only.shape != (512,512):
        print('no 512!')
        label_only = cv2.resize(label_only,(512,512))
    filename = data['imagePath']
    return label_only,filename,label_names

# %%
#EDH,SDH,SAH,ICH,IVH,background
#如果都沒有 最後一張全白 剩下五個都黑的
def seperate_connect_conponent(index,connect_conp):
    mask = np.where(connect_conp == index)
    image = np.zeros((512,512),np.uint8)
    image[mask] = 1
    return image

def make_five_channel_label(label_only,label_names):
    edh = np.zeros((512,512),np.uint8)
    sdh = np.zeros((512,512),np.uint8)
    sah = np.zeros((512,512),np.uint8)
    ich = np.zeros((512,512),np.uint8)
    ivh = np.zeros((512,512),np.uint8)
    none = np.zeros((512,512),np.uint8)
    if len(label_names) == 1 :
        none = np.ones((512,512),np.uint8)
        label = np.stack((edh,sdh,sah,ich,ivh,none),axis = -1)
    else:
        for index,typer in enumerate(label_names):
            if typer != '_background_':
                if typer == 'ICH':
                    ich = seperate_connect_conponent(index,label_only)
                elif typer == 'SAH':
                    sah = seperate_connect_conponent(index,label_only)
                elif typer == 'SDH':
                    sdh = seperate_connect_conponent(index,label_only)
                elif typer == 'EDH':
                    edh = seperate_connect_conponent(index,label_only)
                elif typer == 'IVH':
                    ivh = seperate_connect_conponent(index,label_only)
        label = np.stack((edh,sdh,sah,ich,ivh,none),axis = -1)

    print(label.shape)
    return label

def make_five_channel_label_empty():
    edh = np.zeros((512,512),np.uint8)
    sdh = np.zeros((512,512),np.uint8)
    sah = np.zeros((512,512),np.uint8)
    ich = np.zeros((512,512),np.uint8)
    ivh = np.zeros((512,512),np.uint8)
    none = np.ones((512,512),np.uint8)
    label = np.stack((edh,sdh,sah,ich,ivh,none),axis = -1)
    print(label.shape)
    return label

def filename_cheaker_json(filename):
    pattern = re.compile(r'^.*?.json$')
    match = pattern.match(filename)
    if match:
        return True
    else:
        return False

def filename_cheaker_png(filename):
    pattern = re.compile(r'^.*?.png$')
    match = pattern.match(filename)
    if match:
        return True
    else:
        return False

def load_one_image_with_five_label(image_folder_path):
    original_all_image = []
    original_all_label = []
    #image_has_segmentation = []
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
                print(image_path,' is not 512x512')
                continue
            
            label_file_name = image_file_name.split('.')[0] + '.json'
            label_path = os.path.join(image_folder_path , label_file_name)
            #if json file is exist means that we have this slice's segmentation
            #otherwise we make a blank label(image is all black) for this slice
            if(os.path.isfile(label_path)):
                label_only,filename,label_names = generate_label_png(label_path)
                label = make_five_channel_label(label_only,label_names)
            else:
                label = make_five_channel_label_empty()
            
            print('image size ', image.shape)
            print('label size ', label.shape)
            original_all_image.append(image)
            original_all_label.append(label)
    return original_all_image,original_all_label


def make_labels(base_path):
    #base_path = "/Users/alvinhuang/Desktop/T53-T73_original_file"
    labels = []
    for index,folder_name in enumerate(sorted(listdir(base_path))):
        #print(index,folder_name)
        if folder_name != '.DS_Store':
            image_folder_path = os.path.join(base_path,folder_name)
            for index,image_name in enumerate(sorted(listdir(image_folder_path))):
                if filename_cheaker_png(image_name):
                    label_filename = image_name.split('.')[0] + '.json'
                    index  = int(image_name.split('.')[0][-2:])-1
                    label_path = os.path.join(image_folder_path,label_filename)
                    if os.path.exists(label_path):
                        label_only,filename,label_names = generate_label_png(label_path)
                        label = make_five_channel_label(label_only,label_names)
                    else:
                        label = make_five_channel_label_empty()
                    #print(one_row)
                    labels.append(label)
    labels_array = np.asarray(labels).astype('uint8')
    return labels_array

#%%
def save_one_person_npy(image_folder_path,patient_number):
    base_path = os.path.join('/Users/alvinhuang/Desktop/five_label/old_data_test',patient_number)
    original_all_image,original_all_label = load_one_image_with_five_label(image_folder_path)
    filename = patient_number + '_image'
    filepath = save_to_npy(base_path,filename,np.asarray(original_all_image))
    print(len(original_all_image))
    print('image done! npz in '+ filepath)
    
    filename = patient_number + '_five_label'
    filepath = save_to_npy(base_path,filename,np.asarray(original_all_label))
    print(len(original_all_label))
    print('label done! npz in '+ filepath)

#%%
base_path = "/Users/alvinhuang/Desktop/T01-T21_original_data"
for index,folder_name in zip(range(0,len(listdir(base_path))),sorted(listdir(base_path))):
    print(index,folder_name)
    if folder_name != '.DS_Store':
        image_folder_path = os.path.join(base_path,folder_name)
        print(image_folder_path)
        save_one_person_npy(image_folder_path,folder_name)


#%%
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
    filename = patient_number + '_five_label'
    filepath = save_to_npy(base_path,filename,np.asarray(image))

# %%
labels_array = make_labels("/Users/alvinhuang/Desktop/T01-T21_original_data")
print(labels_array.shape)

# %%
save_one_person_npy('old_data_train',labels_array)