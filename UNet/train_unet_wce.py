import os
import sys
import time
import numpy as np
import pandas as pd
import datetime

import wandb
import torch
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
import albumentations as A

from utils.datasetLADD import LADD
from utils.dataloader import collater_mask, ToTorch, Augmenter, Normalizer
from unet import model
from loss import Weighted_Cross_Entropy_Loss, MSE, IOU_loss


# import torch.autograd
# torch.autograd.set_detect_anomaly(True)


epochs = 30
batch_size = 8 # не хватает видеопамяти при батче 16 даже на (640х640)
num_workers = 8
weights_name = '1_unet_wce_full_data_step_5'
path_to_save = '/home/maantonov_1/VKR/weights/unet/cross/23_02_2024/' + weights_name

#optimazer
start_lr   = 0.0005
num_steps  = 5
gamma_coef = 0.5

os.environ["WANDB_MODE"]="offline" 
wandb.init(project="VKR", entity="matvey_antonov", name = f"{weights_name}_{datetime.date.today().isoformat()}")

local_transform = A.Compose([ # flip inside augmenter
    A.GaussNoise(p=0.2), 
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.3)
])


train_df = [{'dataframe': pd.read_csv('/home/maantonov_1/VKR/data/main_data/crop_train/croped_main_train.csv'),
             'image_dir': '/home/maantonov_1/VKR/data/main_data/crop_train/images'},
            {'dataframe': pd.read_csv('/home/maantonov_1/VKR/data/small_train/full_crop_train/croped_small_train.csv'),
             'image_dir': '/home/maantonov_1/VKR/data/small_train/full_crop_train/images'}]

valid_df = [{'dataframe': pd.read_csv('/home/maantonov_1/VKR/data/main_data/crop_val/croped_val.csv'),
             'image_dir': '/home/maantonov_1/VKR/data/main_data/crop_val/images'}]

train_dataset = LADD(train_df, mode = "train", class_mask=True, transforms = T.Compose([Augmenter(local_transform), Normalizer(), ToTorch()]))
valid_dataset = LADD(valid_df, mode = "valid", class_mask=True, transforms = T.Compose([Normalizer(), ToTorch()]))

print(f'dataset Created', flush=True)

train_data_loader = DataLoader(
    train_dataset,
    batch_size = batch_size,
    shuffle = True,
    num_workers = num_workers,
    collate_fn = collater_mask
)

valid_data_loader = DataLoader(
    valid_dataset,
    batch_size = batch_size,
    shuffle = True,
    num_workers = num_workers,
    collate_fn = collater_mask
)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()
print('CUDA available: {}'.format(torch.cuda.is_available()), flush=True)
print(f'Creating model ===>', flush=True)

model = model.UNet(n_classes=2)

optimizer = torch.optim.Adam(model.parameters(), lr = start_lr)
criterion = Weighted_Cross_Entropy_Loss()

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = num_steps, gamma=gamma_coef)
# lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, factor=0.5, patience=5, verbose=True
# )

model.to(device)


def train_one_epoch(epoch_num, train_data_loader):
    
    print("Train Epoch - {} Started".format(epoch_num), flush=True)
    st = time.time()
    
    model.train()
    
    epoch_loss = []

    for iter_num, data in enumerate(train_data_loader):
                
        optimizer.zero_grad()
        
        img, mask = data['img'].to(device).float(), data['mask'].to(device).long()
        

        pred = model(img)
        loss = criterion(pred, mask)

        if bool(loss == 0):
            continue
                
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    
        optimizer.step()

        epoch_loss.append(float(loss))
            
        print(
            'Epoch: {} | Iteration: {} Map loss {:1.5f}| Running loss: {:1.5f}'.format(
                epoch_num, iter_num, float(loss) , np.mean(epoch_loss)), flush=True)
        
        del loss

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
            img, mask = data['img'].to(device).float(), data['mask'].to(device).long()
            pred = model(img)
            loss = criterion(pred, mask)
            epoch_loss.append(float(loss))

            print(
            'Epoch: {} | Iteration: {} Map loss {:1.5f}| Running loss: {:1.5f}'.format(
                epoch_num, iter_num, float(loss) , np.mean(epoch_loss)), flush=True)
        

            del loss
        
    et = time.time()
    print("\n Total Time - {}\n".format(int(et - st)))
    
    # lr_scheduler.step(np.mean(epoch_loss))

    torch.save(model, f"{path_to_save}_{epoch_num}_{np.mean(epoch_loss)}.pt")
    return np.mean(epoch_loss)
    
    
for epoch in range(epochs):
    st = time.time()
    mean_loss_train = train_one_epoch(epoch, train_data_loader)
    
    mean_loss_val = valid_one_epoch(epoch, valid_data_loader)
    et = time.time()
    
    wandb.log({"epoch": epoch, "train_loss": float(mean_loss_train), 
                   "val_loss": float(mean_loss_val),  "total_time":int(et - st)})