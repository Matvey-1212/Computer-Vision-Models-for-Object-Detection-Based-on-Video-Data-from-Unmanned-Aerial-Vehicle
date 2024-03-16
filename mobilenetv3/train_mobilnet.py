import os
import time
import numpy as np
import pandas as pd
import datetime

import wandb
import torch
import torch.optim as optim
import torchvision.transforms as T
from torchvision.utils import make_grid 
from torch.utils.data import DataLoader
import albumentations as A

from utils.datasetLADD import LADD
from utils.dataloader import collater_mask, ToTorch, Augmenter, Normalizer, Resizer

from mobilseg.model.segmentation import MobileNetV3Seg
from loss import OAN_Focal_Loss, Weighted_Cross_Entropy_Loss





epochs = 15
batch_size = 32
num_workers = 4
weights_name = 'mobilenet_main_lr=0.0003_step_5'
path_to_save = '/home/maantonov_1/VKR/weights/mobilenet/10_03_2024/' + weights_name

#optimazer
start_lr   = 0.0003
num_steps  = 5
gamma_coef = 0.5

#criterion
wce_weight = 30

os.environ["WANDB_MODE"]="offline" 
wandb.init(project="VKR", entity="matvey_antonov", name = f"{weights_name}_{datetime.date.today().isoformat()}")


local_transform = A.Compose([ # flip inside augmenter
    A.GaussNoise(p=0.2), 
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.4)
])


train_df = [{'dataframe': pd.read_csv('/home/maantonov_1/VKR/data/crope_data/main/crop_train_1024/crop_train_1024.csv'),
             'image_dir': '/home/maantonov_1/VKR/data/crope_data/main/crop_train_1024/images'}]

valid_df = [{'dataframe': pd.read_csv('/home/maantonov_1/VKR/data/crope_data/main/crop_val_1024/crop_val_1024.csv'),
             'image_dir': '/home/maantonov_1/VKR/data/crope_data/main/crop_val_1024/images'}]

train_dataset = LADD(train_df, mode = "train", small_class_mask=True, small_mask_coef = 32, transforms = T.Compose([Augmenter(local_transform), Normalizer(), ToTorch()]))
valid_dataset = LADD(valid_df, mode = "valid", small_class_mask=True, small_mask_coef = 32, transforms = T.Compose([Normalizer(), ToTorch()]))

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

model = MobileNetV3Seg(2, backbone='mobilenetv3_small', pretrained_base=False, aux=False)

optimizer = torch.optim.Adam(model.parameters(), lr = start_lr)
criterion = Weighted_Cross_Entropy_Loss(w_1 = wce_weight)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = num_steps, gamma=gamma_coef)

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
            print(
            'Epoch: {} | Iteration: {} Map loss {:1.5f}| Running loss: {:1.5f}'.format(
                epoch_num, iter_num, float(loss) , np.mean(epoch_loss)), flush=True)
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
    
    torch.save(model, f"{path_to_save}_{epoch_num}_{np.mean(epoch_loss)}.pt")
    return np.mean(epoch_loss)
    
    
for epoch in range(epochs):
    st = time.time()
    mean_loss_train = train_one_epoch(epoch, train_data_loader)
    mean_loss_val = valid_one_epoch(epoch, valid_data_loader)
    et = time.time()
    
    wandb.log({"epoch": epoch, "train_loss": float(mean_loss_train), 
                   "val_loss": float(mean_loss_val),  "total_time":int(et - st)})