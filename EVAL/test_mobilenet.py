import os
import time
import numpy as np
import pandas as pd
import cv2

import wandb
import torch
import torch.optim as optim
import torchvision.transforms as T
from torchvision.utils import make_grid 
from torch.utils.data import DataLoader

from utils.datasetLADD import LADD
from utils.dataloader import collater_mask, ToTorch, Augmenter, Normalizer
from mobilseg.model.segmentation import MobileNetV3Seg

from metrics import calculate_semantic_metrics, iou_pytorch, dice_pytorch






test_df = [{'dataframe': pd.read_csv('/home/maantonov_1/VKR/data/main_data/test/test_main.csv'),
             'image_dir': '/home/maantonov_1/VKR/data/main_data/test/images'}]


test_dataset = LADD(test_df, mode = "valid", small_class_mask=True, small_mask_coef = 32, smart_crop = True, new_shape = (1024,1024), transforms = T.Compose([Normalizer(), ToTorch()]))


print(f'dataset Created', flush=True)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()
print('CUDA available: {}'.format(torch.cuda.is_available()), flush=True)
print(f'Crreating retinanet ===>', flush=True)

path_to_weights = '/home/maantonov_1/VKR/weights/mobilenet/06_03_2024/mobilenet_main_lr=0.0003_step_5_14_0.010929136687228757.pt'

# model = OAN.OAN();

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

time_running = []

for i in range(len(test_dataset)):
    print(f'{i} / {len(test_dataset)}')
    
    
            
    with torch.no_grad():
        
        data = test_dataset[i]
        
        img, mask = data['img'].float(), data['mask'].long()
        img = img.permute(2, 0, 1).to(device).float().unsqueeze(dim=0)
        
        # print(f'img {img.shape}')
        t = time.time()
        pred = model(img)
        t1 = time.time()
        
        time_running.append(t1-t)
        print(f'    time: {t1-t}')
        
        pred = torch.argmax(pred, dim=1)[0]
        
        pred = pred.cpu()
        # print(f'pred {pred.shape}')
        # print(f'mask {mask.shape}')
        # exit()
        # torch.save(pred, f'/home/maantonov_1/VKR/actual_scripts/EVAL/pred_img/pred{i}.pt')
        # torch.save(mask, f'/home/maantonov_1/VKR/actual_scripts/EVAL/pred_img/mask{i}.pt')
        # torch.save(img, f'/home/maantonov_1/VKR/actual_scripts/EVAL/pred_img/img{i}.pt')
        
        accuracy, precision, recall, f1 = calculate_semantic_metrics(pred, mask, f_coef = f_coef)
        
        accuracy_mean.append(accuracy)
        precision_mean.append(precision)
        recall_mean.append(recall)
        f1_mean.append(f1)
        
        iou.append(iou_pytorch(pred, mask))
        dice.append(dice_pytorch(pred, mask))
        


        

    
print(f'Accuracy: {np.mean(accuracy_mean)}')
print(f'Precision: {np.mean(precision_mean)}')
print(f'Recall: {np.mean(recall_mean)}')
print(f'F{f_coef} Score: {np.mean(f1_mean)}')
print(f'iou: {np.mean(iou)}')
print(f'dice: {np.mean(dice)}')
print(f'AVG time: {np.mean(time_running)}')