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
from utils.dataloader import collater, collater2, ToTorch, Augmenter, Normalizer, UnNormalizer, AddDim, collater_mosaic
from retinanet import model
from retinanet import OAN
from loss import OAN_Focal_Loss, Weighted_Cross_Entropy_Loss

import albumentations as A

os.environ["WANDB_MODE"]="offline" 
wandb.init(project="VKR", entity="matvey_antonov", name = "resnet_oan_main3")



local_transform = A.Compose([
    # A.HorizontalFlip(p=0.5),  # Случайное отражение по горизонтали с вероятностью 0.5
    # A.VerticalFlip(p=0.5),  # Случайное отражение по вертикали с вероятностью 0.5
    A.RandomBrightnessContrast(p=0.2),  # Случайное изменение яркости/контрастности с вероятностью 0.2ы
    A.GaussNoise(p=0.1),  # Добавление гауссовского шума с вероятностью 0.1
    
    # Цветовые аугментации
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.3),  # Случайное изменение яркости, контраста, насыщенности и оттенка
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),  # Изменение оттенка, насыщенности и яркости
])




train_df = [{'dataframe': pd.read_csv('/home/maantonov_1/VKR/data/main_data/crop_train/croped_main_train.csv'),
             'image_dir': '/home/maantonov_1/VKR/data/main_data/crop_train/images'}]
            # ,
            # {'dataframe': pd.read_csv('/home/maantonov_1/VKR/data/small_train/full_crop_train/croped_small_train.csv'),
            #  'image_dir': '/home/maantonov_1/VKR/data/small_train/full_crop_train/images'}]

valid_df = [{'dataframe': pd.read_csv('/home/maantonov_1/VKR/data/main_data/crop_val/croped_val.csv'),
             'image_dir': '/home/maantonov_1/VKR/data/main_data/crop_val/images'}]

train_dataset = LLAD(train_df, mode = "train", small_class_mask=True, transforms = T.Compose([Augmenter(local_transform), Normalizer(), ToTorch()]))
valid_dataset = LLAD(valid_df, mode = "valid", small_class_mask=True, transforms = T.Compose([Normalizer(), ToTorch()]))

print(f'dataset Created', flush=True)

# DataLoaders
train_data_loader = DataLoader(
    train_dataset,
    batch_size = 32,
    shuffle = True,
    num_workers = 8,
    collate_fn = collater
)

valid_data_loader = DataLoader(
    valid_dataset,
    batch_size = 32,
    shuffle = True,
    num_workers = 8,
    collate_fn = collater
)



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()
print('CUDA available: {}'.format(torch.cuda.is_available()), flush=True)
print(f'Crreating retinanet ===>', flush=True)

# retinanet = model.resnet50(num_classes = 1, pretrained = False)
oan = OAN.OAN();

optimizer = torch.optim.Adam(oan.parameters(), lr = 0.0003)
criterion = Weighted_Cross_Entropy_Loss(w_1 = 10)

# Learning Rate Scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma=0.5)

# retinanet.to(device)
oan.to(device)

#No of epochs
epochs = 30

def train_one_epoch(epoch_num, train_data_loader):
    
    print("Train Epoch - {} Started".format(epoch_num), flush=True)
    st = time.time()
    
    oan.train()
    
    epoch_loss = []

    for iter_num, data in enumerate(train_data_loader):
                
        # Reseting gradients after each iter
        optimizer.zero_grad()
        
        img, mask = data['img'].to(device).float(), data['mask'].to(device).long()
        
       
        pred = oan(img)
        
        loss = criterion(pred, mask)
        

        if bool(loss == 0):
            continue
                
        # Calculating Gradients
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(oan.parameters(), 0.1)
                
        # Updating Weights
        optimizer.step()

        #Epoch Loss
        epoch_loss.append(float(loss))

            
        print(
            'Epoch: {} | Iteration: {} Map loss {:1.5f}| Running loss: {:1.5f}'.format(
                epoch_num, iter_num, float(loss) , np.mean(epoch_loss)), flush=True)
        
        del loss
        
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
            img, mask = data['img'].to(device).float(), data['mask'].to(device).long()
        
       
            pred = oan(img)
            
            loss = criterion(pred, mask)


            #Epoch Loss
            epoch_loss.append(float(loss))

            print(
            'Epoch: {} | Iteration: {} Map loss {:1.5f}| Running loss: {:1.5f}'.format(
                epoch_num, iter_num, float(loss) , np.mean(epoch_loss)), flush=True)
        

            del loss
        
    et = time.time()
    print("\n Total Time - {}\n".format(int(et - st)))
    
    # Save Model after each epoch
    torch.save(oan, f"resnet_oan_main3_{epoch_num}_{np.mean(epoch_loss)}.pt")
    return np.mean(epoch_loss)
    
    
for epoch in range(epochs):
    st = time.time()
# Call train function
    mean_loss_train = train_one_epoch(epoch, train_data_loader)
    
    mean_loss_val = valid_one_epoch(epoch, valid_data_loader)
    et = time.time()
    
    wandb.log({"epoch": epoch, "train_loss": float(mean_loss_train), 
                   "val_loss": float(mean_loss_val),  "total_time":int(et - st)})