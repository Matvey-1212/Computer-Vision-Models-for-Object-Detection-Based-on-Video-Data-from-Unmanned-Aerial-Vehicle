import os
import time
import numpy as np
import pandas as pd
import cv2
import datetime


import wandb
import torch
import torch.optim as optim
import torchvision.transforms as T
from torchvision.utils import make_grid 
from torch.utils.data import DataLoader

from utils.datasetLADD import LADD
from utils.dataloader import collater_mask,  ToTorch, Augmenter, Normalizer
from unet.networks import R2AttU_Net
from metrics import calculate_semantic_metrics, iou_pytorch, dice_pytorch






test_df = [{'dataframe': pd.read_csv('/home/maantonov_1/VKR/data/main_data/test/test_main.csv'),
             'image_dir': '/home/maantonov_1/VKR/data/main_data/test/images'}]


test_dataset = LADD(test_df, mode = "valid", class_mask=True, smart_crop = True, new_shape = (2048,2048), transforms = T.Compose([Normalizer(), ToTorch()]))


print(f'dataset Created', flush=True)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()
print('CUDA available: {}'.format(torch.cuda.is_available()), flush=True)
print(f'Crreating retinanet ===>', flush=True)

path_to_weights = '/home/maantonov_1/VKR/weights/at_unet_main/main/2024-03-15/wce_weight60/2024-03-15_at_unet_main_lr_lr0.0003_step5_wce_weight60_n:14_iou:0.6158406734466553_f:0.25592119318719303_val:0.2326761195411072_last.pt'


model = torch.load(path_to_weights)


# if torch.cuda.is_available():
#     model = torch.nn.DataParallel(model).cuda()

model.to(device)
model.training = False
model.eval()



time_running = []

accuracy_mean = []
precision_mean = []
recall_mean = []
f1_mean = []
iou = []
dice = []

f_coef = 0.5

for i in range(len(test_dataset)):
    print(f'{i} / {len(test_dataset)}')
    
    
            
    with torch.no_grad():
        
        data = test_dataset[i]
        
        img, mask = data['img'].float(), data['mask'].long()
        t = time.time()
        img = img.permute(2, 0, 1).to(device).float().unsqueeze(dim=0)
        t1 = time.time()
        
        time_running.append(t1-t)
        print(f'    time: {t1-t}')
        
        # print(f'img {img.shape}')
        
        pred = model(img)
        pred = torch.argmax(pred, dim=1)[0]
        
        pred = pred.cpu()
        
        
        
        accuracy, precision, recall, f1 = calculate_semantic_metrics(pred, mask, f_coef = f_coef)
        
        accuracy_mean.append(accuracy)
        precision_mean.append(precision)
        recall_mean.append(recall)
        f1_mean.append(f1)
        
        iou.append(iou_pytorch(pred, mask))
        dice.append(dice_pytorch(pred, mask))
        


        

print(f'{datetime.date.today().isoformat()}')
print(f'Accuracy: {np.mean(accuracy_mean)}')
print(f'Precision: {np.mean(precision_mean)}')
print(f'Recall: {np.mean(recall_mean)}')
print(f'F{f_coef} Score: {np.mean(f1_mean)}')
print(f'iou: {np.mean(iou)}')
print(f'dice: {np.mean(dice)}')
print(f'AVG time: {np.mean(time_running)}')