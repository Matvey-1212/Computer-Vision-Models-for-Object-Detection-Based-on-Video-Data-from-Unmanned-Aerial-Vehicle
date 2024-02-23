# check list
# 1) wandb
# 2) dataset config
# 3) dataloader config
# 4) model config
# 5) weight name

import os
import time
import numpy as np
import pandas as pd

import wandb
import torch
import torch.optim as optim
import torchvision.transforms as T

from torch.utils.data import DataLoader

from utils.datasetLLAD_for_SSD import LLAD
from utils.dataloader import collater, collater2, ToTorch, Augmenter, Normalizer, UnNormalizer, AddDim, collater_without_mask

from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from vision.ssd.ssd import MatchPrior
from vision.ssd.vgg_ssd import create_vgg_ssd
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.ssd.mobilenetv3_ssd_lite import create_mobilenetv3_large_ssd_lite, create_mobilenetv3_small_ssd_lite
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite

from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import vgg_ssd_config
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.config import squeezenet_ssd_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform

# import torch.autograd
# torch.autograd.set_detect_anomaly(True)




# os.environ["WANDB_MODE"]="offline" 
# wandb.init(project="VKR", entity="matvey_antonov", name='train_vgg')

DIR_TRAIN = '/home/maantonov_1/VKR/data/main_data/train/images'
train_df = pd.read_csv('/home/maantonov_1/VKR/data/main_data/train/train_main.csv')
valid_df = pd.read_csv('/home/maantonov_1/VKR/data/main_data/train/val_main.csv')

train_dataset = LLAD(train_df, DIR_TRAIN, mode = "train", smart_crop = True, gaus = False, class_mask=False, new_shape = (640, 640), transforms = T.Compose([Normalizer(), ToTorch()]))
valid_dataset = LLAD(valid_df, DIR_TRAIN, mode = "valid", smart_crop = True, gaus = False, class_mask=False, new_shape = (640, 640), transforms = T.Compose([Normalizer(), ToTorch()]))

print(f'dataset Created', flush=True)

# DataLoaders
train_data_loader = DataLoader(
    train_dataset,
    batch_size = 8,
    shuffle = True,
    num_workers = 4,
    collate_fn = collater_without_mask
)

valid_data_loader = DataLoader(
    valid_dataset,
    batch_size = 8,
    shuffle = True,
    num_workers = 4,
    collate_fn = collater_without_mask
)



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()
print('CUDA available: {}'.format(torch.cuda.is_available()), flush=True)
print(f'Crreating retinanet ===>', flush=True)

## num_classes
num_classes = 1

## 'vgg16-ssd'
# create_net = create_vgg_ssd
# config = vgg_ssd_config

# ## 'mb1-ssd'
create_net = create_mobilenetv1_ssd
config = mobilenetv1_ssd_config

# ## 'mb1-ssd-lite'
# create_net = create_mobilenetv1_ssd_lite
# config = mobilenetv1_ssd_config

# ## 'sq-ssd-lite'
# create_net = create_squeezenet_ssd_lite
# config = squeezenet_ssd_config

# ## 'mb2-ssd-lite'
# create_net = lambda num: create_mobilenetv2_ssd_lite(num, width_mult=args.mb2_width_mult)
# config = mobilenetv1_ssd_config

# ## 'mb3-large-ssd-lite'
# create_net = lambda num: create_mobilenetv3_large_ssd_lite(num)
# config = mobilenetv1_ssd_config

# ## 'mb3-small-ssd-lite'
# create_net = lambda num: create_mobilenetv3_small_ssd_lite(num)
# config = mobilenetv1_ssd_config


model = create_net(num_classes)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3, center_variance=0.1, size_variance=0.2, device=device)

# Learning Rate Scheduler
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma=0.5)


model.to(device)

#No of epochs
epochs = 15

def train_one_epoch(epoch_num, train_data_loader, debug_steps = 100):
    
    print("Train Epoch - {} Started".format(epoch_num), flush=True)
    st = time.time()
    
    model.train(True)
    epoch_loss = []
    running_regression_loss = 0.0
    running_classification_loss = 0.0

    for iter_num, data in enumerate(train_data_loader):
                
        images = data['img']
        # images, boxes, labels = data['img'], data['']
        images = images.to(device)
        # boxes = boxes.to(device)
        # labels = labels.to(device)

        optimizer.zero_grad()
        confidence, locations = model(images)
        
        print(f'confidence {confidence.shape}')
        print(f'confidence {confidence}')
        print(f'locations {locations.shape}')
        print(f'locations {locations}')
        exit()
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())
        
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
                
        
            
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            confidence, locations = model(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss
        

        epoch_loss.append(loss.item())

        
        print(
            'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(epoch_loss)), flush=True)
        
        del loss
        
    et = time.time()
    print("\n Total Time - {}\n".format(int(et - st)))
    
    # Save Model after each epoch
    torch.save(model, f"ssd_{np.mean(epoch_loss)}.pt")
    return np.mean(epoch_loss)
    
    
for epoch in range(epochs):
    st = time.time()
# Call train function
    mean_loss_train = train_one_epoch(epoch, train_data_loader)
    
    mean_loss_val = valid_one_epoch(epoch, valid_data_loader)
    et = time.time()
    
    wandb.log({"epoch": epoch, "train_loss": float(mean_loss_train), 
                   "val_loss": float(mean_loss_val),  "total_time":int(et - st)})