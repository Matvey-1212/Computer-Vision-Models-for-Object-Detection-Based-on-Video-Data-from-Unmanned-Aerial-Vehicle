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
from utils.dataloader import collater, ToTorch, Augmenter, Normalizer, UnNormalizer, AddDim
from unet import UNet
# from metric import iou, pix_acc
from loss import Weighted_Cross_Entropy_Loss, MSE, IOU_loss

# import torch.autograd
# torch.autograd.set_detect_anomaly(True)




# os.environ["WANDB_MODE"]="offline" 
# wandb.init(project="VKR", entity="matvey_antonov")

DIR_TRAIN = '/home/maantonov_1/VKR/data/main_data/train/images'
train_df = pd.read_csv('/home/maantonov_1/VKR/data/main_data/train/train_main.csv')
valid_df = pd.read_csv('/home/maantonov_1/VKR/data/main_data/train/val_main.csv')

train_dataset = LLAD(train_df, DIR_TRAIN, mode = "train", smart_crop = True, gaus = True, class_mask=False, new_shape = (640, 640), transforms = T.Compose([Normalizer(), ToTorch()]))
valid_dataset = LLAD(valid_df, DIR_TRAIN, mode = "valid", smart_crop = True, gaus = True, class_mask=False, new_shape = (640, 640), transforms = T.Compose([Normalizer(), ToTorch()]))

print(f'dataset Created', flush=True)

# DataLoaders
train_data_loader = DataLoader(
    train_dataset,
    batch_size = 12,
    shuffle = True,
    num_workers = 4,
    collate_fn = collater
)

valid_data_loader = DataLoader(
    valid_dataset,
    batch_size = 12,
    shuffle = True,
    num_workers = 4,
    collate_fn = collater
)



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()
print('CUDA available: {}'.format(torch.cuda.is_available()), flush=True)
print(f'Crreating retinanet ===>', flush=True)

# retinanet = model.resnet50(num_classes = 1, pretrained = False)
model = UNet(n_classes=1)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.0003)
criterion = MSE()

# Learning Rate Scheduler
#lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma=0.5)

# retinanet.to(device)
model.to(device)

#No of epochs
epochs = 15

def train_one_epoch(epoch_num, train_data_loader):
    
    print("Train Epoch - {} Started".format(epoch_num), flush=True)
    st = time.time()
    
    model.train()
    
    epoch_loss = []

    for iter_num, data in enumerate(train_data_loader):
                
        # Reseting gradients after each iter
        optimizer.zero_grad()
        
        img, mask = data['img'].cuda().float(), data['mask'].cuda().float()
        
        
        # img = oan(img)
        
        # Forward
        pred = model(img)
        loss = criterion(pred, mask)

        if bool(loss == 0):
            continue
                
        # Calculating Gradients
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                
        # Updating Weights
        optimizer.step()

        #Epoch Loss
        epoch_loss.append(float(loss))

            
        print(
            'Epoch: {} | Iteration: {} Map loss {:1.5f}| Running loss: {:1.5f}'.format(
                epoch_num, iter_num, float(loss) , np.mean(epoch_loss)), flush=True)
        
        del loss
        
    # Update the learning rate
    #if lr_scheduler is not None:
        #lr_scheduler.step()
        
    et = time.time()
    print("\n Total Time - {}\n".format(int(et - st)))
    return np.mean(epoch_loss)
        

    
def valid_one_epoch(epoch_num, valid_data_loader):
    
    print("Val Epoch - {} Started".format(epoch_num))
    st = time.time()
    
    epoch_loss = []

    for iter_num, data in enumerate(valid_data_loader):
                
        with torch.no_grad():
            
            # Forward
            img, mask = data['img'].cuda().float(), data['mask'].cuda().float()
        
    
        # img = oan(img)
        
        # Forward
            pred = model(img)
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
    torch.save(model, f"unet_mse_{np.mean(epoch_loss)}.pt")
    return np.mean(epoch_loss)
    
    
for epoch in range(epochs):
    st = time.time()
# Call train function
    mean_loss_train = train_one_epoch(epoch, train_data_loader)
    
    mean_loss_val = valid_one_epoch(epoch, valid_data_loader)
    et = time.time()
    
    wandb.log({"epoch": epoch, "train_loss": float(mean_loss_train), 
                   "val_loss": float(mean_loss_val),  "total_time":int(et - st)})