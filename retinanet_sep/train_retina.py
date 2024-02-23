import os
import time
import numpy as np
import pandas as pd

import wandb
import torch
import torch.optim as optim
import torchvision.transforms as T
from torchvision.utils import make_grid 
from torch.utils.data import DataLoader

from utils.datasetLLAD import LLAD
from retinanet import model
from utils.dataloader import collater, ToTorch, Augmenter, Normalizer, UnNormalizer, AddDim, collater_annot
from retinanet import OAN

import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import random
from torchvision.transforms import RandomApply, GaussianBlur

import albumentations as A

# import torch.autograd
# torch.autograd.set_detect_anomaly(True)

local_transform = A.Compose([ # flip inside augmenter
    # A.RandomBrightnessContrast(p=0.2),  # Случайное изменение яркости/контрастности с вероятностью 0.2ы
    A.GaussNoise(p=0.1),  # Добавление гауссовского шума с вероятностью 0.1
    
    # Цветовые аугментации
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.3),  # Случайное изменение яркости, контраста, насыщенности и оттенка
    # A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),  # Изменение оттенка, насыщенности и яркости
])



os.environ["WANDB_MODE"]="offline" 
wandb.init(project="VKR", entity="matvey_antonov", name = "just_retina_full_640_step_7")

train_df = [{'dataframe': pd.read_csv('/home/maantonov_1/VKR/data/main_data/crop_train/croped_main_train.csv'),
             'image_dir': '/home/maantonov_1/VKR/data/main_data/crop_train/images'},
            {'dataframe': pd.read_csv('/home/maantonov_1/VKR/data/small_train/full_crop_train/croped_small_train.csv'),
             'image_dir': '/home/maantonov_1/VKR/data/small_train/full_crop_train/images'}]

valid_df = [{'dataframe': pd.read_csv('/home/maantonov_1/VKR/data/main_data/crop_val/croped_val.csv'),
             'image_dir': '/home/maantonov_1/VKR/data/main_data/crop_val/images'}]

train_dataset = LLAD(train_df, mode = "train", transforms = T.Compose([Augmenter(local_transform), Normalizer(), ToTorch()]))
valid_dataset = LLAD(valid_df, mode = "valid", transforms = T.Compose([Normalizer(), ToTorch()]))



print(f'dataset Created', flush=True)

# DataLoaders
train_data_loader = DataLoader(
    train_dataset,
    batch_size = 32,
    shuffle = True,
    num_workers = 8,
    collate_fn = collater_annot
)

valid_data_loader = DataLoader(
    valid_dataset,
    batch_size = 32,
    shuffle = True,
    num_workers = 8,
    collate_fn = collater_annot
)



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()
print('CUDA available: {}'.format(torch.cuda.is_available()), flush=True)
print(f'Crreating retinanet ===>', flush=True)



retinanet = model.resnet50(num_classes = 1, pretrained = False, inputs = 3)

optimizer = optim.Adam(retinanet.parameters(), lr=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma=0.5)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, verbose=True)

retinanet.to(device)



#No of epochs
epochs = 25

def train_one_epoch(epoch_num, train_data_loader):
    
    print("Train Epoch - {} Started".format(epoch_num), flush=True)
    st = time.time()
    
    retinanet.train()
    
    epoch_loss = []

    for iter_num, data in enumerate(train_data_loader):
                
        optimizer.zero_grad()
        
        img, annot = data['img'].cuda().float(), data['annot'].cuda().float()
        
        classification_loss, regression_loss = retinanet([img, annot])
                
        classification_loss = classification_loss.mean()
        regression_loss = regression_loss.mean()
        

        loss = classification_loss + regression_loss
        

        if bool(loss == 0):
            continue
                
        loss.backward()

        torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
                
        optimizer.step()
        epoch_loss.append(float(loss))

            
        print(
            'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(epoch_loss)), flush=True)
        
        del classification_loss
        del regression_loss
        
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
            
        
            classification_loss, regression_loss = retinanet([img, annot])
                    
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            

            loss = classification_loss + regression_loss

            epoch_loss.append(float(loss))

            print(
            'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(epoch_loss)), flush=True)
        
        del classification_loss
        del regression_loss
        
    et = time.time()
    print("\n Total Time - {}\n".format(int(et - st)))
    

    torch.save(retinanet, f"/home/maantonov_1/VKR/actual_scripts/retinanet_sep/temp_weights_retina/retinanet_only_{epoch_num}_{np.mean(epoch_loss)}.pt")
    return np.mean(epoch_loss)
    
    
for epoch in range(epochs):
    st = time.time()

    mean_loss_train = train_one_epoch(epoch, train_data_loader)
    
    mean_loss_val = valid_one_epoch(epoch, valid_data_loader)
    et = time.time()
    
    wandb.log({"epoch": epoch, "train_loss": float(mean_loss_train), 
                   "val_loss": float(mean_loss_val),  "total_time":int(et - st)})