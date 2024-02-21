import os
import time
import numpy as np
import cv2
import pandas as pd

import wandb
import torch
import torch.optim as optim
import torchvision.transforms as T
from torchvision.utils import make_grid 
from torch.utils.data import DataLoader

from UTILS.datasetLLAD import LLAD
from UTILS.dataloader import collater, collater2, ToTorch, Augmenter, Normalizer, UnNormalizer, AddDim
from unet import UNet
# from metric import iou, pix_acc
from UNET.loss import Weighted_Cross_Entropy_Loss, MSE, IOU_loss
from metrics import evaluate

# import torch.autograd
# torch.autograd.set_detect_anomaly(True)




# os.environ["WANDB_MODE"]="offline" 
# wandb.init(project="VKR", entity="matvey_antonov")

DIR_TRAIN = '/home/maantonov_1/VKR/data/main_data/test/images'
train_df = pd.read_csv('/home/maantonov_1/VKR/data/main_data/test/test_main.csv')


test_dataset = LLAD(train_df, DIR_TRAIN, mode = "train", smart_crop = True, gaus = True, class_mask=False, new_shape = (640, 640), transforms = T.Compose([Normalizer(), ToTorch()]))


print(f'dataset Created', flush=True)

# DataLoaders
test_data_loader = DataLoader(
    test_dataset,
    batch_size = 1,
    shuffle = True,
    num_workers = 2,
    collate_fn = collater
)




device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()
print('CUDA available: {}'.format(torch.cuda.is_available()), flush=True)
print(f'Crreating retinanet ===>', flush=True)


criterion = Weighted_Cross_Entropy_Loss()

# model = UNet(n_classes=2)
model = torch.load('/home/maantonov_1/VKR/actual_scripts/UNet-PyTorch/unet_mse_4.135797038173905e-08.pt',  map_location=device)




# Learning Rate Scheduler
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma=0.5)


model.to(device)
model.eval()


for iter_num, data in enumerate(test_data_loader):
            
    with torch.no_grad():
        
        if torch.cuda.is_available():
            img, mask = data['img'].cuda().float(), data['mask'].cuda()
        else:
            img, mask = data['img'].float(), data['mask']
    

    # img = oan(img)

    # Forward
        pred = model(img)
        # loss = criterion(pred, mask)
        # a, b, c, d = img.shape
        # pred.permute(0, 3, 2, 1).view(c,d,a).numpy()
        
        # cv2.imwrite('/home/maantonov_1/VKR/actual_scripts/EVAL/pred_img/img.jpg', img.permute(0, 3, 2, 1).view(c,d,b).numpy()) 
        # cv2.imwrite('/home/maantonov_1/VKR/actual_scripts/EVAL/pred_img/mask.jpg', mask.permute(0, 3, 2, 1).view(c,d,1).numpy()) 
        # cv2.imwrite('/home/maantonov_1/VKR/actual_scripts/EVAL/pred_img/pred.jpg', pred.permute(0, 3, 2, 1).view(c,d,1).numpy()) 
        torch.save(img, '/home/maantonov_1/VKR/actual_scripts/EVAL/pred_img/img.pt')
        torch.save(mask, '/home/maantonov_1/VKR/actual_scripts/EVAL/pred_img/mask.pt')
        torch.save(pred, '/home/maantonov_1/VKR/actual_scripts/EVAL/pred_img/pred.pt')
        break
        
    
    
    
