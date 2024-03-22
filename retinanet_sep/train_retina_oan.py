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
from utils.metrics import evaluate
from retinanet import model_oan
from retinanet import losses

# import torch.autograd
# torch.autograd.set_detect_anomaly(True)



epochs = 30
batch_size = 16
num_workers = 2
oan_gamma = 2
oan_alpha = 0.25
weights_name = 'retinanet_oan_vis_lr0.0003_step_5_gamma:2_alpha:0.25'
path_to_save = '/home/maantonov_1/VKR/weights/retinanet/visdrone/' 

if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)
path_to_save = path_to_save + weights_name

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


# train_df = [{'dataframe': pd.read_csv('/home/maantonov_1/VKR/data/crope_data/main/crop_train_1024/crop_train_1024.csv'),
#              'image_dir': '/home/maantonov_1/VKR/data/crope_data/main/crop_train_1024/images'},
#             {'dataframe': pd.read_csv('/home/maantonov_1/VKR/data/crope_data/main/small_crop_train/crop_small_train.csv'),
#              'image_dir': '/home/maantonov_1/VKR/data/crope_data/main/small_crop_train/images'}]

# valid_df = [{'dataframe': pd.read_csv('/home/maantonov_1/VKR/data/crope_data/main/crop_val_1024/crop_val_1024.csv'),
#              'image_dir': '/home/maantonov_1/VKR/data/crope_data/main/crop_val_1024/images'}]

# train_dataset = LADD(train_df, mode = "train", transforms = T.Compose([Augmenter(local_transform), Normalizer(), ToTorch()]))
# valid_dataset = LADD(valid_df, mode = "valid", transforms = T.Compose([Normalizer(), ToTorch()]))

train_df = [{'dataframe': pd.read_csv('/home/maantonov_1/VKR/data/visdrone/visdrone/VisDrone2019-DET-train/visdrone_train.csv'),
             'image_dir': '/home/maantonov_1/VKR/data/visdrone/visdrone/VisDrone2019-DET-train/images'}]

valid_df = [{'dataframe': pd.read_csv('/home/maantonov_1/VKR/data/visdrone/visdrone/VisDrone2019-DET-val/visdrone_val.csv'),
             'image_dir': '/home/maantonov_1/VKR/data/visdrone/visdrone/VisDrone2019-DET-val/images'}]

train_dataset = LADD(train_df, mode = "train", img_endwith = '.jpg', transforms = T.Compose([Augmenter(local_transform), Normalizer(), Resizer()]))
valid_dataset = LADD(valid_df, mode = "valid",  img_endwith = '.jpg', transforms = T.Compose([Normalizer(), Resizer()]))
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

retinanet = model_oan.resnet50(num_classes = 2, pretrained = False, inputs = 3)
retinanet.focalLoss = losses.FocalLoss(alpha = oan_alpha, gamma = oan_gamma)
retinanet.oan_loss = losses.OANFocalLoss(alpha = oan_alpha, gamma = oan_gamma)

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
        new_elements = torch.where(annot[:, :, 3] == -1, -1, 1)  
        annot[:,:,4] = new_elements
        # print(annot.shape)
        # for i in annot:
        #     print(i)
        # exit()
        
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
            new_elements = torch.where(annot[:, :, 3] == -1, -1, 1)  
            annot[:,:,4] = new_elements
        
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
    

    return np.mean(epoch_loss)

def get_metrics_one_epoch(epoch_num, valid_data_loader, best_f):
    
    print("Val Epoch - {} Started".format(epoch_num))
    st = time.time()
    retinanet.eval()

    epoch_loss = []
    prediction = []

    for iter_num, data in enumerate(valid_data_loader):
                
        with torch.no_grad():
            
            img, annot = data['img'].cuda().float(), data['annot'].cpu().float()
            
            new_elements = torch.where(annot[:, :, 3] == -1, -1, 1)  
            annot[:,:,4] = new_elements
        
            scores, labels, boxes = retinanet(img)
                    
        pred_dict = {}
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy() * 0
        boxes  = boxes.cpu().numpy()
        pred_dict['scores'] = scores
        pred_dict['labels'] = labels
        pred_dict['boxes']  = boxes
        
        annot = annot[annot[:,:, 3] != -1]  
        gt_dict = {}
        if annot.shape[0] != 0:
            gt_labels = annot[:,-1].numpy() * 0
            gt_boxes  = annot[:,:-1].numpy()
        else:
            gt_labels = np.array([])
            gt_boxes  = np.array([])
        gt_dict['boxes'] = gt_boxes
        gt_dict['labels'] = gt_labels
        
        
        

        prediction.append((gt_dict, pred_dict))

            
    map_score, Fscore = evaluate(prediction)
    
    print(
            'Epoch: {} | map_score loss: {:1.5f} | Fscore loss: {:1.5f} '.format(
                epoch_num, float(map_score), float(Fscore)), flush=True)
        
    et = time.time()
    print("\n Total Time - {}\n".format(int(et - st)))
    

    if best_f < map_score:
        best_f = map_score
        torch.save(retinanet, f"{path_to_save}_n{epoch_num}_m:{map_score:0.2f}_f:{Fscore:0.2f}.pt")
    elif epoch_num >= epochs - 1:
        torch.save(retinanet, f"{path_to_save}_n{epoch_num}_m:{map_score:0.2f}_f:{Fscore:0.2f}_last.pt")
        
    return map_score, Fscore, best_f
    
best_f = 0
    
for epoch in range(epochs):
    st = time.time()

    mean_loss_train = train_one_epoch(epoch, train_data_loader)
    
    mean_loss_val = valid_one_epoch(epoch, valid_data_loader)
    
    map_score, Fscore, best_f = get_metrics_one_epoch(epoch, valid_data_loader, best_f)
    et = time.time()
    
    wandb.log({"epoch": epoch, "train_loss": float(mean_loss_train), "val_loss": float(mean_loss_val), 
                   "map_score": float(map_score), "Fscore": float(Fscore), "total_time":int(et - st)})