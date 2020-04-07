import numpy as np
import os, cv2, random, sys ,re, pydicom
import matplotlib.pyplot as plt
import matplotlib
import albumentations as albu
import segmentation_models as sm
import keras
from matplotlib.colors import ListedColormap
from os import listdir

## for prevent out of memory (suit for computer center)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
## Please put model and this program in the same folder
## In the same computer or envioriment, the pretrained model from keras just need to download one time.
def main():
    #folder_or_file = sys.argv[1]
    #types = sys.argv[2]
    location = sys.argv[1]
    #print(folder_or_file)
    print(location)
    #folder_name = location.split('/')[-1]
    folder_name = location.split('\\')[-1]
    #folder_name=re.split('\\ | /',location)[-1]
    #print(folder_name)

    #if types == 'dicom':
    x_test,image_filenames = load_dicom_from_folder(location)
    #elif types == 'png':
    #    x_test,image_filenames = load_png_from_folder(location)
    
    test_dataset =  test_Dataset(x_test,classes=['background', 'ich'],augmentation=get_test_augmentation())
    predict_data = get_predict_data(test_dataset)
    predict_data = make_to_3dimension_data(x_test)

    print("Start to load pretrained model......")
    model = sm.Unet('densenet121',activation='sigmoid',classes=1)
    print("Finish to load pretrained model!")

    print("Start to load model weights......")
    MODEL_FULL_NAME = "2020_03_09_08_52_23_unet_densenet121_png_albu_dia_bceadddice_1510sample.hdf5"
    model.load_weights(MODEL_FULL_NAME)
    print("Finish to load model weights!")

    print("Start to predict......")
    predict = model.predict(predict_data,batch_size=8)
    print("Finish to predict!")

    result = rule_judge(predict)
    print("Result : ",result)

    filename = folder_name + "_result_image.png"
    predict_binary = np.where(predict>0.5,1,0)
    visualize_data_with_label(x_test,predict_binary,filename)

    print("Start to make mask for one slices.....")
    make_mask_per_slides(x_test,predict,image_filenames)

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

def load_png_from_folder(image_folder_path):
    original_all_image = []
    image_file_list = listdir(image_folder_path)
    image_file_list = [ name for name in image_file_list if filename_cheaker_png(name) == True ]
    image_file_list = sorted(image_file_list)
    #print(image_file_list)

    for image_file_name in image_file_list:
        #read file that is .png
        if(filename_cheaker_png(image_file_name) == True):
            image_path = os.path.join(image_folder_path , image_file_name)
            image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)

            if (image.shape != (512,512)):
                image = cv2.resize(image,(512,512))

            original_all_image.append(image)
    train_folder = np.asarray(original_all_image).astype('uint8')
    return train_folder,image_file_list

# classes for data loading and preprocessing
class test_Dataset:
    CLASSES = ['background', 'ich']
    def __init__(self,images_npy,classes=None, augmentation=None, preprocessing=None):
        self.images = images_npy
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):

        image = cv2.cvtColor(self.images[i],cv2.COLOR_GRAY2RGB)
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        return image

    def __len__(self):
        return len(self.images)

def get_predict_data(test_dataset):
    predict_data = []
    for index in range(0,len(test_dataset)):
        image = test_dataset[index]
        predict_data.append(image)
    predict_data = np.asarray(predict_data)
    #print("predict_data shape: ",predict_data.shape)
    return predict_data

def get_test_augmentation():
    train_transform = [
        albu.augmentations.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=True, p=1.0)
    ]
    return albu.Compose(train_transform)

def make_to_3dimension_data(npy_image_array):
    new_3d = []
    image_aug = augmentation()
    for one_image in npy_image_array:
      new_3d_image = cv2.cvtColor(one_image,cv2.COLOR_GRAY2RGB)
      new_image = image_aug(image = new_3d_image)['image']
      new_3d.append(new_image)
    return np.asarray(new_3d)

def augmentation():
    train_transform = [
        albu.augmentations.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=True, p=1.0)
    ]
    return albu.Compose(train_transform)

#metric for evaluation
def dice_cof(label_gt,predict):
  #print(label_gt_abs,predict_abs)
  #to binary
  predict_binary = np.where(predict.flatten() > 0.5,1,0)
  label_binary = np.where(label_gt.flatten() > 0.5,1,0)

  label_gt_abs = len(np.where(label_binary >0)[0])
  predict_abs = len(np.where(predict_binary>0)[0])
  #print(label_gt_abs,predict_abs)
  intersection = np.array(cv2.bitwise_and(label_binary,predict_binary),dtype=np.uint8)
  intersection_abs = len(np.where(intersection > 0)[0])
  #print(label_gt_abs,predict_abs,intersection_abs)
  dice = ((2*intersection_abs)) / (label_gt_abs + predict_abs + 0.001)
  #print(round(dice,3))
  return dice

def rule_judge(predict):
  #do binary to the mask
  #check the segmentation percentage to one mask
  prob_threshold = 0.5
  segmentation_percentage_of_the_mask = 0.01
  person_cheaker = []
  for one_predict_img in predict:
      one_predict_img = one_predict_img.flatten()
      one_predict_img = np.where(one_predict_img>prob_threshold,1,0)
      percentage = len(np.where(one_predict_img>0)[0]) / len(np.where(one_predict_img))
      if percentage > segmentation_percentage_of_the_mask:
          person_cheaker.append(1)
      else:
          person_cheaker.append(0)
  #print('person_cheaker: ',person_cheaker)

  #count the dice between two mask
  neighbor_dice = []
  for index in range(0,len(person_cheaker)-1):
      neighbor_dice.append(round(dice_cof(predict[index],predict[index+1]),3))
  #print('neighbor_dice: ', neighbor_dice)

  #get the none zero sequences that in between dice sequence  
  sublist = []
  flag = 0
  one_sub=[]
  for index in range(0,len(neighbor_dice)):
      if neighbor_dice[index]!=0 and flag == 0:
          one_sub.append(neighbor_dice[index])
          flag = 0
      elif neighbor_dice[index] ==0 and flag == 0:
          sublist.append(one_sub)
          one_sub = []
          flag = 1
      elif neighbor_dice[index] ==0 and flag != 0:
          pass
      elif neighbor_dice[index] !=0 and flag != 0:
          one_sub.append(neighbor_dice[index])
          flag = 0
  #print('sublist: ',sublist)

  #use the sequence to check if the subsequence's length is bigger than 2(3 masks)
  #and one of dice value in the subsequence is bigger than 0.3 just judge it has ich
  dice_sensitive =0.15
  sublist_len_limit = 1
  #one number in sublen is two image's dice means that the continue is set to 2
  judge = 0
  for index in range(0,len(sublist)):
      pair_dice_len = len([pair_dice for pair_dice in sublist[index] if pair_dice >= dice_sensitive])
      if pair_dice_len>0 and len(sublist[index])>=sublist_len_limit :
          judge = 1
  if judge == 1:
      result = "True"
  else:
      result = "False"
  
  return result

def visualize_data_with_label(data,label,filename):
    color_list = ["none", "red" ]
    nan_background = ListedColormap(color_list)
    num_of_data = len(data)
    if num_of_data%10 !=0 :
        row = int(num_of_data / 10) + 1
    else:
        row = int(num_of_data / 10)
    #print(row)
    plt.figure(figsize=(10*2,row*2))
    for index in range(0,len(data)):
        plt.subplot(row,10,index+1)
        plt.axis('off')
        plt.imshow(data[index].reshape(512,512),cmap='gray')
        plt.imshow(label[index].reshape(512,512),cmap = nan_background,alpha = 0.3)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(filename)
    plt.close()

def visualize_two_image_compare(original_image,predict_image,index,filename):
    cmap_no_background = matplotlib.colors.LinearSegmentedColormap.from_list("", ["none","blue", 'cyan', 'green', 'yellow', 'red'])
    #img must be a array
    plt.figure(figsize=(15,7))
    plt.subplot(1,2,1)
    plt.axis('off')
    plt.title('original '+ str(index))
    plt.imshow(original_image.reshape(512,512),cmap='gray')
    plt.colorbar(shrink=0.8,orientation='vertical',pad = 0.05)

    plt.subplot(1,2,2)
    plt.axis('off')
    plt.title('Predict '+ str(index) + '/ Max_prob:'+str(round(predict_image.max(),2)))
    plt.imshow(original_image.reshape(512,512),cmap='gray')
    plt.imshow(predict_image.reshape(512,512),cmap = cmap_no_background,alpha=0.3)
    plt.colorbar(shrink=0.8,orientation='vertical',pad = 0.05)

    max_prob = round(predict_image.max(),3)
    filename = filename.split(".")[0]
    
    final_filename = filename + '_1_'+str(max_prob)+'.png'
    plt.savefig(final_filename,bbox_inches='tight')
    plt.close()

def make_mask_per_slides(x_test,predict,image_filenames):
    for original_image,pimage,index,filename in zip(x_test,predict,range(1,len(x_test)+1),image_filenames):
        if len(np.where(pimage > 0.5)[0]) > 1 :
            visualize_two_image_compare(original_image,pimage,index,filename)
            print("Make "+filename+" mask ok!")

if __name__ == '__main__':
    main()
