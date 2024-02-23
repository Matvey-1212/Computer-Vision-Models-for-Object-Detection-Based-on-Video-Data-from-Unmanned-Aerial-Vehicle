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

from retinanet.datasetLLAD import LLAD
from retinanet import model
from retinanet.dataloader import collater, ToTorch, Augmenter, Normalizer, UnNormalizer, AddDim
from retinanet import OAN

# import torch.autograd
# torch.autograd.set_detect_anomaly(True)




os.environ["WANDB_MODE"]="offline" 
wandb.init(project="VKR", entity="matvey_antonov", name = "just_retina_main_640")

DIR_TRAIN = '/home/maantonov_1/VKR/data/main_data/train/images'
train_df = pd.read_csv('/home/maantonov_1/VKR/data/main_data/train/train_main.csv')
valid_df = pd.read_csv('/home/maantonov_1/VKR/data/main_data/train/val_main.csv')

train_dataset = LLAD(train_df, DIR_TRAIN, mode = "train", smart_crop = True, gaus = False, class_mask=False, new_shape = (640, 640), transforms = T.Compose([Normalizer(), ToTorch()]))
valid_dataset = LLAD(valid_df, DIR_TRAIN, mode = "valid", smart_crop = True, gaus = False, class_mask=False, new_shape = (640, 640), transforms = T.Compose([Normalizer(), ToTorch()]))



print(f'dataset Created', flush=True)

# DataLoaders
train_data_loader = DataLoader(
    train_dataset,
    batch_size = 16,
    shuffle = True,
    num_workers = 8,
    collate_fn = collater
)

valid_data_loader = DataLoader(
    valid_dataset,
    batch_size = 16,
    shuffle = True,
    num_workers = 8,
    collate_fn = collater
)



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()
print('CUDA available: {}'.format(torch.cuda.is_available()), flush=True)
print(f'Crreating retinanet ===>', flush=True)



retinanet = model.resnet50(num_classes = 1, pretrained = False, inputs = 3)

optimizer = torch.optim.Adam(retinanet.parameters(), lr = 0.0001)

# Learning Rate Scheduler
#lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma=0.5)

retinanet.to(device)



#No of epochs
epochs = 1

def train_one_epoch(epoch_num, train_data_loader):
    
    print("Train Epoch - {} Started".format(epoch_num), flush=True)
    st = time.time()
    
    retinanet.train()
    
    epoch_loss = []

    for iter_num, data in enumerate(train_data_loader):
                
        # Reseting gradients after each iter
        optimizer.zero_grad()
        
        img, annot = data['img'].cuda().float(), data['annot'].cuda().float()
        
        
        
        classification_loss, regression_loss = retinanet([img, annot])
                
        # Calculating Loss
        classification_loss = classification_loss.mean()
        regression_loss = regression_loss.mean()
        

        loss = classification_loss + regression_loss
        

        if bool(loss == 0):
            continue
                
        # Calculating Gradients
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
                
        # Updating Weights
        optimizer.step()

        #Epoch Loss
        epoch_loss.append(float(loss))

            
        print(
            'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(epoch_loss)), flush=True)
        
        del classification_loss
        del regression_loss
        
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
            
            img, annot = data['img'].cuda().float(), data['annot'].cuda().float()
            
        
            classification_loss, regression_loss = retinanet([img, annot])
                    
            # Calculating Loss
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            

            loss = classification_loss + regression_loss

            #Epoch Loss
            epoch_loss.append(float(loss))

            print(
            'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(epoch_loss)), flush=True)
        
        del classification_loss
        del regression_loss
        
    et = time.time()
    print("\n Total Time - {}\n".format(int(et - st)))
    
    # Save Model after each epoch
    torch.save(retinanet, f"retinanet_only_{int(time.time())}_{np.mean(epoch_loss)}.pt")
    return np.mean(epoch_loss)
    
    
for epoch in range(epochs):
    st = time.time()

    mean_loss_train = train_one_epoch(epoch, train_data_loader)
    
    mean_loss_val = valid_one_epoch(epoch, valid_data_loader)
    et = time.time()
    
    wandb.log({"epoch": epoch, "train_loss": float(mean_loss_train), 
                   "val_loss": float(mean_loss_val),  "total_time":int(et - st)})