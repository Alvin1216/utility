#%%
import labelme
from labelme import utils
import os, cv2, random, re , json, pydicom
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
    #label_only = utils.shapes_to_label((512,512), data['shapes'], label_name_to_value).astype('uint8')
    label_only = utils.shapes_to_label((512,512), data['shapes'], label_name_to_value)[0].astype('uint8')
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

#%%
generate_label_png('C:\\Users\\alvinhuang\\Desktop\\test\\ICH0081\\ICH0081VS20.json')
# %%
#EDH,SDH,SAH,ICH,IVH,background
#如果都沒有 最後一張全白 剩下五個都黑的
def seperate_connect_conponent(index,connect_conp):
    mask = np.where(connect_conp == index)
    image = np.zeros((512,512),np.uint8)
    image[mask] = 1
    return image

def background_maker(connect_conp):
    mask = np.where(connect_conp >= 1)
    image = np.ones((512,512),np.uint8)
    image[mask] = 0
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
        none = background_maker(label_only)
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

def filename_cheaker_dcm(filename):
    pattern = re.compile(r'^.*?.dcm$')
    match = pattern.match(filename)
    if match:
        return True
    else:
        return False

def transform_to_hu(medical_image, image):
    intercept = medical_image.RescaleIntercept
    slope = medical_image.RescaleSlope
    hu_image = image * slope + intercept
    return hu_image

def window_image(image, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max    
    return window_image

def load_dicom_from_folder(image_folder_path):
    original_all_image = []
    image_file_list = listdir(image_folder_path)
    image_file_list = [ name for name in image_file_list if filename_cheaker_dcm(name) == True ]
    image_file_list = sorted(image_file_list)
    #print(image_file_list)

    for image_file_name in image_file_list:
        #read file that is .png
        if(filename_cheaker_dcm(image_file_name) == True):
            image_path = os.path.join(image_folder_path , image_file_name)
            medical_image = pydicom.read_file(image_path)
            image_dicom = medical_image.pixel_array
            hu_image = transform_to_hu(medical_image,image_dicom)
            brain_image = window_image(hu_image, 35, 90)
            brain_image_normalized = (((brain_image-brain_image.min())/(brain_image.max()-brain_image.min()))*255).astype('uint8')

            if (brain_image_normalized.shape != (512,512)):
                image = cv2.resize(brain_image_normalized,(512,512))
            else:
                image = brain_image_normalized

            original_all_image.append(image)
    train_folder = np.asarray(original_all_image).astype('uint8')
    return train_folder,image_file_list

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
    print(base_path)
    labels = []
    images = []
    dicom_images = []
    single_labels = []
    markers = []
    # for index,folder_name in enumerate(sorted(listdir(base_path))):
    #     #print(index,folder_name)
    #     if folder_name != '.DS_Store':
    #         image_folder_path = os.path.join(base_path,folder_name)
    for index,image_name in enumerate(sorted(listdir(base_path))):
        if filename_cheaker_png(image_name):
            print(image_name)

            #load_png
            image_path = os.path.join(base_path , image_name)
            image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
            if (image.shape != (512,512)):
                image = cv2.resize(image,(512,512))
                print(image_path,' is not 512x512')
                continue
            
            #load_dicom
            #dicom_filename = image_name.split('.')[0] + '.dcm'
            #dicom_filename = filename.split('-')[1] + '.dcm'
            filename = image_name.split('.')[0]
            dicom_filename = image_name.split('.')[0] + '.dcm'
            dicom_path = os.path.join(base_path , dicom_filename)
            medical_image = pydicom.read_file(dicom_path)
            image_dicom = medical_image.pixel_array
            hu_image = transform_to_hu(medical_image,image_dicom)
            brain_image = window_image(hu_image, 35, 90)
            brain_image_normalized = (((brain_image-brain_image.min())/(brain_image.max()-brain_image.min()))*255).astype('uint8')

            if (brain_image_normalized.shape != (512,512)):
                dicom_image = cv2.resize(brain_image_normalized,(512,512))
                print(dicom_path,' dicom_image is not 512x512')
            else:
                dicom_image = brain_image_normalized

            #load_five_label
            label_filename = image_name.split('.')[0] + '.json'
            index  = int(image_name.split('.')[0][-2:])-1
            label_path = os.path.join(base_path,label_filename)
            if os.path.exists(label_path):
                label_only,filename,label_names = generate_label_png(label_path)
                label = make_five_channel_label(label_only,label_names)
                single_label = np.ones((512,512)) - label[:,:,-1]
                marker = 1
            else:
                label = make_five_channel_label_empty()
                single_label = np.ones((512,512)) - label[:,:,-1]
                marker = 0

            labels.append(label)
            images.append(image)
            dicom_images.append(dicom_image)
            single_labels.append(single_label)
            markers.append(marker)
    labels_array = np.asarray(labels).astype('uint8')
    images_array = np.asarray(images).astype('uint8')
    dicom_images_array = np.asarray(dicom_images).astype('uint8')
    #single_labels_array = np.asarray(single_labels).astype('uint8')
    return labels_array,images_array,dicom_images_array,single_labels,markers

#%%
def save_one_person_npy(image_folder_path,patient_number):
    base_path = os.path.join('/Users/alvinhuang/Desktop/170207',patient_number)
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
base_path = "/Users/alvinhuang/Desktop/ICH171-207"
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

# def save_one_person_npy(patient_number,image):
#     base_path = os.path.join('/Users/alvinhuang/Desktop/ich_dicom_numpy',patient_number)
#     filename = patient_number + '_five_label'
#     filepath = save_to_npy(base_path,filename,np.asarray(image))

# %%
#C:\Users\alvinhuang\Desktop\ICH081-ICH170\ICH0081
base_path = "C:\\Users\\alvinhuang\\Desktop\\ICH171-207"

for index,folder_name in zip(range(0,len(listdir(base_path))),sorted(listdir(base_path))):
    save_numpy_path = "C:\\Users\\alvinhuang\\Desktop\\171207"
    save_new_folder = os.path.join(save_numpy_path,folder_name)
    if not os.path.exists(save_new_folder):
        os.makedirs(save_new_folder)
    
    source_folder = os.path.join(base_path,folder_name)
    print(source_folder)
    labels_array,images_array,dicom_images_array,single_labels,makers = make_labels(source_folder)
    print('labels_array shape : ',labels_array.shape)
    print('images_array(png) shape : ',images_array.shape)
    print('dicom_images_array shape : ',dicom_images_array.shape)
    #print('single_labels shape : ',len(single_labels))
    print('makers shape : ',len(makers))
    save_to_npy(save_new_folder,folder_name+'_5labels.npy',labels_array)
    save_to_npy(save_new_folder,folder_name+'_png_img.npy',images_array)
    save_to_npy(save_new_folder,folder_name+'_dicom_img.npy',dicom_images_array)
    #save_to_npy(save_new_folder,folder_name+'_single_label.npy',single_labels)
    save_to_npy(save_new_folder,folder_name+'_makers.npy',makers)

# %%
save_one_person_npy('old_data_train',labels_array)

#%%
def plot_6_channel(one_images):
  #one_images = one_images*255
  plt.figure(figsize=(10,5))
  for index in range(0,6):
    #print(index)
    #predict_binary = np.where(one_images[:,:,index]>0.5,1,0)
    plt.subplot(1,6,index+1)
    plt.imshow(one_images[:,:,index],vmin = 0,vmax = 1)
    plt.axis("off")

# %%
