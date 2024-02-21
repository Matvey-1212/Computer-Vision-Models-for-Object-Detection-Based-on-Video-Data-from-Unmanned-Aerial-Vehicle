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

from utils.datasetLLAD import LLAD
from utils.dataloader import collater, collater2, ToTorch, Augmenter, Normalizer, UnNormalizer, AddDim, collater_mosaic
import OAN
# from metric import iou, pix_acc
from metrics import calculate_semantic_metrics, iou_pytorch, dice_pytorch






test_df = [{'dataframe': pd.read_csv('/home/maantonov_1/VKR/data/main_data/test/test_main.csv'),
             'image_dir': '/home/maantonov_1/VKR/data/main_data/test/images'}]


test_dataset = LLAD(test_df, mode = "valid", small_class_mask=True, smart_crop = True, new_shape = (1024,1024), transforms = T.Compose([Normalizer(), ToTorch()]))


print(f'dataset Created', flush=True)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()
print('CUDA available: {}'.format(torch.cuda.is_available()), flush=True)
print(f'Crreating retinanet ===>', flush=True)



model = OAN.OAN();

model.load_state_dict(torch.load('/home/maantonov_1/VKR/actual_scripts/resnet encoder/resnet_oan_main3_27_0.05811988045611689.pt').state_dict())


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

for i in range(len(test_dataset)):
    print(f'{i} / {len(test_dataset)}')
    
    
            
    with torch.no_grad():
        
        data = test_dataset[i]
        
        img, mask = data['img'].float(), data['mask'].long()
        img = img.permute(2, 0, 1).to(device).float().unsqueeze(dim=0)
        
        # print(f'img {img.shape}')
        
        pred = model(img)
        pred = torch.argmax(pred, dim=1)[0]
        
        pred = pred.cpu()
        # print(f'pred {pred.shape}')
        # print(f'mask {mask.shape}')
        # exit()
        # torch.save(pred, f'/home/maantonov_1/VKR/actual_scripts/EVAL/pred_img/pred{i}.pt')
        # torch.save(mask, f'/home/maantonov_1/VKR/actual_scripts/EVAL/pred_img/mask{i}.pt')
        # torch.save(img, f'/home/maantonov_1/VKR/actual_scripts/EVAL/pred_img/img{i}.pt')
        
        accuracy, precision, recall, f1 = calculate_semantic_metrics(pred, mask)
        
        accuracy_mean.append(accuracy)
        precision_mean.append(precision)
        recall_mean.append(recall)
        f1_mean.append(f1)
        
        iou.append(iou_pytorch(pred, mask))
        dice.append(dice_pytorch(pred, mask))
        


        

    
print(f'Accuracy: {np.mean(accuracy_mean)}')
print(f'Precision: {np.mean(precision_mean)}')
print(f'Recall: {np.mean(recall_mean)}')
print(f'F1 Score: {np.mean(f1_mean)}')
print(f'iou: {np.mean(iou)}')
print(f'dice: {np.mean(dice)}')