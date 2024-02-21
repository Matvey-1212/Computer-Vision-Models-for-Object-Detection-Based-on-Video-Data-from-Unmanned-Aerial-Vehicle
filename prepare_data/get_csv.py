import os
import cv2
from PIL import Image
import pandas as pd
import numpy as np

labels_path = '/home/maantonov_1/VKR/data/small_train/croped_train_data/train/labels'
img_path    = '/home/maantonov_1/VKR/data/small_train/croped_train_data/train/images'

llist = []
count = 0

def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size  

for filename in os.listdir(img_path):
    combined_content = ""

    if filename.endswith('.jpg'):
        
        file_path = os.path.join(img_path, filename)
        try:
            weidth, height = get_image_size(file_path)
            
            with open(labels_path+'/'+filename.replace('jpg','txt'), 'r') as file:
                lines = file.readlines()
                for i in lines:
                    img_list = []
                    img_list.append(int(filename.replace('.jpg','')))
                    img_list.append(weidth)
                    img_list.append(height)
                    if i.rstrip('\n').split(' ')[0] != '0':
                        img_list = img_list + ['']*5
                    else:
                        img_list = img_list + i.rstrip('\n').split(' ')
                    llist.append(img_list)
                if len(lines)==0:
                    llist.append([int(filename.replace('.jpg','')),weidth,height,'','','','',''])
                print('!')
        except:
            print(f'error {img_path+"/"+filename.replace("txt","jpg")}')
            
        
        
    count+=1
    
data = pd.DataFrame(llist, columns = ['id','width', 'height','class','x','y','w','h'])
data = data.sort_values('id').reset_index(drop=True)
data.to_csv('croped_small_train.csv',index=False)