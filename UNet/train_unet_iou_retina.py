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
from unet import UNet
from retinanet import model
# from metric import iou, pix_acc
from loss import Weighted_Cross_Entropy_Loss, MSE, IOU_loss

# import torch.autograd
# torch.autograd.set_detect_anomaly(True)




os.environ["WANDB_MODE"]="offline" 
wandb.init(project="VKR", entity="matvey_antonov", name = "unet_iou_without_relu_retina")

DIR_TRAIN = '/home/maantonov_1/VKR/data/main_data/train/images'
train_df = pd.read_csv('/home/maantonov_1/VKR/data/main_data/train/train_main.csv')
valid_df = pd.read_csv('/home/maantonov_1/VKR/data/main_data/train/val_main.csv')

train_dataset = LLAD(train_df, DIR_TRAIN, mode = "train", smart_crop = True, gaus = False, class_mask=True, new_shape = (640, 640), transforms = T.Compose([Normalizer(), ToTorch()]))
valid_dataset = LLAD(valid_df, DIR_TRAIN, mode = "valid", smart_crop = True, gaus = False, class_mask=True, new_shape = (640, 640), transforms = T.Compose([Normalizer(), ToTorch()]))

print(f'dataset Created', flush=True)

# DataLoaders
train_data_loader = DataLoader(
    train_dataset,
    batch_size = 8,
    shuffle = True,
    num_workers = 8,
    collate_fn = collater
)

valid_data_loader = DataLoader(
    valid_dataset,
    batch_size = 8,
    shuffle = True,
    num_workers = 8,
    collate_fn = collater
)



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()
print('CUDA available: {}'.format(torch.cuda.is_available()), flush=True)
print(f'Crreating retinanet ===>', flush=True)



model1 = UNet(n_classes=2)
optimizer1 = torch.optim.Adam(model1.parameters(), lr = 0.0001)
criterion = Weighted_Cross_Entropy_Loss()

model2 = model.resnet50(num_classes = 1, pretrained = False, inputs = 3)
optimizer2 = torch.optim.Adam(model2.parameters(), lr = 0.0001)

# Learning Rate Scheduler
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma=0.5)


model1.to(device)
model2.to(device)

#No of epochs
epochs = 15

def train_one_epoch(epoch_num, train_data_loader):
    
    print("Train Epoch - {} Started".format(epoch_num), flush=True)
    st = time.time()
    
    model1.train()
    model2.train()
    
    epoch_loss1 = []
    epoch_loss2 = []

    for iter_num, data in enumerate(train_data_loader):
                
        img, mask, annot = data['img'].cuda().float(), data['mask'].cuda().long(), data['annot'].cuda().float()
        
        pred = model1(img)
        loss1 = criterion(pred, mask)

        # if bool(loss1 == 0):
        #     continue
        
        loss1.backward()
        optimizer1.step()
        epoch_loss1.append(float(loss1))
        optimizer1.zero_grad()
        
        print(
            'Epoch: {} | Iteration: {} Map loss {:1.5f}| Running loss: {:1.5f}'.format(
                epoch_num, iter_num, float(loss1) , np.mean(epoch_loss1)),end ='', flush=True)
                
        
        classification_loss, regression_loss = model2([img, annot])
                
        classification_loss = classification_loss.mean()
        regression_loss = regression_loss.mean()
        
        loss2 = classification_loss + regression_loss
        
        # if bool(loss2 == 0):
        #     continue
        
        loss2.backward()
        torch.nn.utils.clip_grad_norm_(model2.parameters(), 0.1)
        optimizer2.step()
        epoch_loss2.append(float(loss2))
        optimizer2.zero_grad()
        
        print(
            'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(epoch_loss2)), flush=True)
        print()
        
        
        del loss1
        del loss2
        
    # Update the learning rate
    #if lr_scheduler is not None:
        #lr_scheduler.step()
        
    et = time.time()
    print("\n Total Time - {}\n".format(int(et - st)))
    return np.mean(epoch_loss1), np.mean(epoch_loss2)
        

    
def valid_one_epoch(epoch_num, valid_data_loader):
    
    print("Val Epoch - {} Started".format(epoch_num))
    st = time.time()
    
    epoch_loss1 = []
    epoch_loss2 = []

    for iter_num, data in enumerate(valid_data_loader):
                
        with torch.no_grad():
            
            # Forward
            img, mask, annot = data['img'].cuda().float(), data['mask'].cuda().long(), data['annot'].cuda().float()
        
            pred = model1(img)
            loss1 = criterion(pred, mask)

            epoch_loss1.append(float(loss1))

            print(
            'Epoch: {} | Iteration: {} Map loss {:1.5f}| Running loss: {:1.5f}'.format(
                epoch_num, iter_num, float(loss1) , np.mean(epoch_loss1)),end ='', flush=True)
        

            classification_loss, regression_loss = model2([img, annot])
                
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            loss2 = classification_loss + regression_loss
            epoch_loss2.append(float(loss2))
        
            print(
                'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                    epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(epoch_loss2)), flush=True)
            print()
            
            
            del loss1
            del loss2
        
    et = time.time()
    print("\n Total Time - {}\n".format(int(et - st)))
    
    # Save Model after each epoch
    torch.save(model1, f"unet_iou_without_relu{epoch_num}_{np.mean(epoch_loss1)}.pt")
    torch.save(model2, f"retinanet{epoch_num}_{np.mean(epoch_loss2)}.pt")
    
    return np.mean(epoch_loss1), np.mean(epoch_loss2)
    
    
for epoch in range(epochs):
    st = time.time()
# Call train function
    mean_loss_train1, mean_loss_train2 = train_one_epoch(epoch, train_data_loader)
    
    
    
    mean_loss_val1, mean_loss_val2 = valid_one_epoch(epoch, valid_data_loader)
    et = time.time()
    
    wandb.log({"epoch": epoch, "train_loss_unet": float(mean_loss_train1), "val_loss_unet": float(mean_loss_val1),  "train_loss_retina": float(mean_loss_train2), "val_loss_retina": float(mean_loss_val2), "total_time":int(et - st)})