import labelme
from labelme import utils
import os, cv2, random, re , json
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

def generate_class_type(json_file_path,filename,index,person_name):
    defalt_dic = {'NAME':person_name,'index':index,'FILENAME':filename,'EDH': 0,'SDH' : 0,'SAH': 0,'ICH': 0,'IVH':0,'NONE':1}
    data = json.load(open(json_file_path))
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
    
    for label in label_names:
        if label != '_background_':
            defalt_dic[label] = 1
            defalt_dic['NONE'] = 0
    #print(defalt_dic)
    return defalt_dic

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
    #print(label_names)
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


def make_table(base_path):
    #base_path = "/Users/alvinhuang/Desktop/T53-T73_original_file"
    table = []
    for index,folder_name in zip(range(0,len(listdir(base_path))),sorted(listdir(base_path))):
        #print(index,folder_name)
        if folder_name != '.DS_Store':
            image_folder_path = os.path.join(base_path,folder_name)
            for index,image_name in zip(range(0,len(listdir(image_folder_path))),sorted(listdir(image_folder_path))):
                if filename_cheaker_png(image_name):
                    label_filename = image_name.split('.')[0] + '.json'
                    index  = int(image_name.split('.')[0][-2:])-1
                    label_path = os.path.join(image_folder_path,label_filename)
                    if os.path.exists(label_path):
                        one_row = generate_class_type(label_path,image_name,index,folder_name)
                    else:
                        one_row = {'NAME':folder_name,'index':index,
                                    'FILENAME':image_name,
                                    'EDH': 0,'SDH' : 0,'SAH': 0,'ICH': 0,'IVH':0,'NONE':1}
                    #print(one_row)
                    table.append(one_row)
    return table

table110 = make_table("/Users/alvinhuang/Desktop/ICH0001-80_original_file/ICH0001-10")
table_1121 = make_table("/Users/alvinhuang/Desktop/ICH0001-80_original_file/ICH0011-21")
table_2240 = make_table("/Users/alvinhuang/Desktop/ICH0001-80_original_file/ICH0022-40")
table_4160 = make_table("/Users/alvinhuang/Desktop/ICH0001-80_original_file/ICH0041-60")
table_6180 = make_table("/Users/alvinhuang/Desktop/ICH0001-80_original_file/ICH0061-80")
table = table110 + table_1121 + table_2240 + table_4160 + table_6180

print(len(table))
import csv
with open('ich0001_80.csv', 'w') as f:  # Just use 'w' mode in 3.x
    for index,row in zip(range(0,len(table)),table):
        if index == 0:
            w = csv.DictWriter(f, row.keys())
            w.writeheader()
        else:
            w.writerow(row)