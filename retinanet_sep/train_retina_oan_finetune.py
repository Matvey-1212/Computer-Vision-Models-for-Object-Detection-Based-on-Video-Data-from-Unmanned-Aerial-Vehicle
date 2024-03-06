import os
import time
import numpy as np
import pandas as pd
import random
import datetime

import wandb
import torch
import torch.optim as optim

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms as T
import torchvision.transforms.functional as F
import albumentations as A

from utils.datasetLADD import LADD
from utils.dataloader import collater_annot, ToTorch, Augmenter, Normalizer, Resizer
from retinanet import model_oan

# import torch.autograd
# torch.autograd.set_detect_anomaly(True)



epochs = 15
batch_size = 12
num_workers = 4
weights_name = 'retinanet_oan_vis+small_lr+ful_0.0003_step5'
path_to_save = '/home/maantonov_1/VKR/weights/retinanet/05_03_2024/' + weights_name

path_to_weights = '/home/maantonov_1/VKR/weights/retinanet/05_03_2024/retinanet_oan_vis+small_lr0.0003_step5_14_1.1158647922908558.pt'

#optimazer
start_lr   = 0.0003
num_steps  = 5
gamma_coef = 0.5

os.environ["WANDB_MODE"]="offline" 
wandb.init(project="VKR", entity="matvey_antonov", name = f"{weights_name}_{datetime.date.today().isoformat()}")


local_transform = A.Compose([ # flip inside augmenter
    A.GaussNoise(p=0.2), 
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.3)
])


train_df = [{'dataframe': pd.read_csv('/home/maantonov_1/VKR/data/crope_data/main/crop_train_1024/crop_train_1024.csv'),
             'image_dir': '/home/maantonov_1/VKR/data/crope_data/main/crop_train_1024/images'}]

valid_df = [{'dataframe': pd.read_csv('/home/maantonov_1/VKR/data/crope_data/main/crop_val_1024/crop_val_1024.csv'),
             'image_dir': '/home/maantonov_1/VKR/data/crope_data/main/crop_val_1024/images'}]

train_dataset = LADD(train_df, mode = "train", transforms = T.Compose([Augmenter(local_transform), Normalizer(), ToTorch()]))
valid_dataset = LADD(valid_df, mode = "valid", transforms = T.Compose([Normalizer(), ToTorch()]))

# train_df = [{'dataframe': pd.read_csv('/home/maantonov_1/VKR/data/visdrone/visdrone/VisDrone2019-DET-train/visdrone_train.csv'),
#              'image_dir': '/home/maantonov_1/VKR/data/visdrone/visdrone/VisDrone2019-DET-train/images'}]

# valid_df = [{'dataframe': pd.read_csv('/home/maantonov_1/VKR/data/visdrone/visdrone/VisDrone2019-DET-val/visdrone_val.csv'),
#              'image_dir': '/home/maantonov_1/VKR/data/visdrone/visdrone/VisDrone2019-DET-val/images'}]

# train_dataset = LADD(train_df, mode = "train", img_endwith = '.jpg', transforms = T.Compose([Augmenter(local_transform), Normalizer(), Resizer()]))
# valid_dataset = LADD(valid_df, mode = "valid",  img_endwith = '.jpg', transforms = T.Compose([Normalizer(), Resizer()]))
print(f'dataset Created', flush=True)



train_data_loader = DataLoader(
    train_dataset,
    batch_size = batch_size,
    shuffle = True,
    num_workers = num_workers,
    collate_fn = collater_annot
)

valid_data_loader = DataLoader(
    valid_dataset,
    batch_size = batch_size,
    shuffle = True,
    num_workers = num_workers,
    collate_fn = collater_annot
)



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()
print('CUDA available: {}'.format(torch.cuda.is_available()), flush=True)
print(f'Crreating retinanet ===>', flush=True)

# retinanet = model_oan.resnet50(num_classes = 1, pretrained = False, inputs = 3)

retinanet = torch.load(path_to_weights, map_location=device)

optimizer = optim.Adam(retinanet.parameters(), lr = start_lr)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = num_steps, gamma=gamma_coef)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, verbose=True)

retinanet.to(device)


def train_one_epoch(epoch_num, train_data_loader):
    
    print("Train Epoch - {} Started".format(epoch_num), flush=True)
    st = time.time()
    
    retinanet.train()
    
    epoch_loss = []

    for iter_num, data in enumerate(train_data_loader):
                
        optimizer.zero_grad()
        
        img, annot = data['img'].cuda().float(), data['annot'].cuda().float()
        
        classification_loss, regression_loss, oan_loss, _ = retinanet([img, annot])
                
        classification_loss = classification_loss.mean()
        regression_loss = regression_loss.mean()

        loss = classification_loss + regression_loss + 3 * oan_loss
        

        if bool(loss == 0):
            print(
            'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | oan loss: {:1.5f} | Running loss: {:1.5f}'.format(
                epoch_num, iter_num, float(classification_loss), float(regression_loss), float(oan_loss), np.mean(epoch_loss)), flush=True)
            continue
                
        loss.backward()

        torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
                
        optimizer.step()
        epoch_loss.append(float(loss))

            
        print(
            'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | oan loss: {:1.5f} | Running loss: {:1.5f}'.format(
                epoch_num, iter_num, float(classification_loss), float(regression_loss), float(oan_loss), np.mean(epoch_loss)), flush=True)
        
        del classification_loss
        del regression_loss
        del oan_loss
        
    # Update the learning rate
    if lr_scheduler is not None:
        lr_scheduler.step()
        
    et = time.time()
    print("\n Total Time - {}\n".format(int(et - st)))
    return np.mean(epoch_loss)
        

    
def valid_one_epoch(epoch_num, valid_data_loader):
    
    print("Val Epoch - {} Started".format(epoch_num))
    st = time.time()
    

    epoch_loss = []

    for iter_num, data in enumerate(valid_data_loader):
                
        with torch.no_grad():
            
            img, annot = data['img'].cuda().float(), data['annot'].cuda().float()
            
        
            classification_loss, regression_loss, oan_loss, _ = retinanet([img, annot])
                    
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            

            loss = classification_loss + regression_loss + 3 * oan_loss

            epoch_loss.append(float(loss))

            print(
            'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | oan loss: {:1.5f} | Running loss: {:1.5f}'.format(
                epoch_num, iter_num, float(classification_loss), float(regression_loss), float(oan_loss), np.mean(epoch_loss)), flush=True)
        
        del classification_loss
        del regression_loss
        del oan_loss
        
    et = time.time()
    print("\n Total Time - {}\n".format(int(et - st)))
    

    torch.save(retinanet, f"{path_to_save}_{epoch_num}_{np.mean(epoch_loss)}.pt")
    return np.mean(epoch_loss)
    
    
for epoch in range(epochs):
    st = time.time()

    mean_loss_train = train_one_epoch(epoch, train_data_loader)
    
    mean_loss_val = valid_one_epoch(epoch, valid_data_loader)
    et = time.time()
    
    wandb.log({"epoch": epoch, "train_loss": float(mean_loss_train), 
                   "val_loss": float(mean_loss_val),  "total_time":int(et - st)})