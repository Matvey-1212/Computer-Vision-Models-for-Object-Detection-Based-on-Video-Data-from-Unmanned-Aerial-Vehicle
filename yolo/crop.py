import os
import time
import numpy as np
import pandas as pd

import cv2
import random
import shutil




source_dir1 = '/home/maantonov_1/VKR/data/small_train/croped_train_data/val/images'


c = []
for filename in os.listdir(source_dir1):
    if filename == '.DS_Store':
        continue

    c.append(filename.replace('.jpg',''))

source_dir = '/home/maantonov_1/VKR/data/small_train/croped_train_data/train/labels'
target_dir = '/home/maantonov_1/VKR/data/small_train/croped_train_data/val/labels'

count = 0
for filename in os.listdir(source_dir):
    if filename == '.DS_Store':
        continue
    print(f'{count}')
    count+=1
    if filename.replace('.txt','') in c:
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, filename)
        shutil.move(source_path, target_path)

    
    
    