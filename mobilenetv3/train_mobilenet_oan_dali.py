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

from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

from utils.datasetLADD import LADD
from utils.dataloader import collater_annot, ToTorch, Augmenter, Normalizer, Resizer
from utils.Dali import get_dali_pipeline, get_dali_pipeline_aug, flip_bboxes2, flip_bboxes
from utils.metrics import calculate_semantic_metrics, iou_pytorch, dice_pytorch
from mobilseg.model.segmentation import MobileNetV3Seg,MobileNetV3_custom_fpn
from loss import OAN_Focal_Loss, Weighted_Cross_Entropy_Loss1



# import torch.autograd
# torch.autograd.set_detect_anomaly(True)



epochs = 15
batch_size = 16
num_workers = 2
wce_weight = 5

#optimazer
start_lr   = 0.0005
num_steps  = 15
gamma_coef = 0.5

print(f'epochs {epochs}')
print(f'batch_size {batch_size}')
print(f'num_workers {num_workers}')
print(f'wce_weight {wce_weight}')
print(f'start_lr {start_lr}')
print(f'num_steps {num_steps}')
print(f'gamma_coef {gamma_coef}')

# weights_name = f'{datetime.date.today().isoformat()}_retinanet_oan_vis+small_lr+ful_lr{start_lr}_step{num_steps}_gamma{oan_gamma}_alpha{oan_alpha}'
weights_name = f'{datetime.date.today().isoformat()}_mobilenetv3_large_main_lr_lr{start_lr}_step{num_steps}_wce_weight{wce_weight}'
path_to_save = f'/home/maantonov_1/VKR/weights/mobilenetv3_large/main/{datetime.date.today().isoformat()}/wce_weight{wce_weight}/'
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)
path_to_save = path_to_save + weights_name

print(f'path_to_save {path_to_save}')



os.environ["WANDB_MODE"]="offline" 
wandb.init(project="VKR", entity="matvey_antonov", name = f"{weights_name}")


main_dir = '/home/maantonov_1/VKR/data/crope_data/main/crop_train_1024/'
# main_dir = '/home/maantonov_1/VKR/data/crope_data/small/small_crop_train/'
images_dir_tarin = main_dir + 'images'
annotations_file_train = main_dir + 'annotations/annot.json'

main_dir = '/home/maantonov_1/VKR/data/crope_data/main/crop_val_1024/'
# main_dir = '/home/maantonov_1/VKR/data/crope_data/small/small_crop_val/'
images_dir_val = main_dir + 'images'
annotations_file_val = main_dir + 'annotations/annot.json'


dali_iterator_train = DALIGenericIterator(
    pipelines=[get_dali_pipeline_aug(images_dir = images_dir_tarin, annotations_file = annotations_file_train, batch_size = batch_size, num_threads = num_workers)],
    output_map=['data', 'bboxe', 'bbox_shapes', 'img_id', 'horizontal_flip','vertical_flip'],
    reader_name='Reader',
    last_batch_policy=LastBatchPolicy.PARTIAL,
    auto_reset=True,
    dynamic_shape=True
)

dali_iterator_val = DALIGenericIterator(
    pipelines=[get_dali_pipeline(images_dir = images_dir_val, annotations_file = annotations_file_val, batch_size = batch_size, num_threads = num_workers)],
    output_map=['data', 'bboxe', 'bbox_shapes', 'img_id'],
    reader_name='Reader',
    last_batch_policy=LastBatchPolicy.PARTIAL,
    auto_reset=True,
    dynamic_shape=True
)

print(f'dataset Created', flush=True)




device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()
print('CUDA available: {}'.format(torch.cuda.is_available()), flush=True)
print(f'Crreating model ===>', flush=True)

model = MobileNetV3_custom_fpn(2, backbone='mobilenetv3_large', pretrained_base=False, aux=False, attention = True)

# criterion = Weighted_Cross_Entropy_Loss(w_1 = wce_weight)

criterion1 = OAN_Focal_Loss()
criterion2 = Weighted_Cross_Entropy_Loss1(w_1 = wce_weight)

optimizer = optim.AdamW(model.parameters(), lr = start_lr)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = num_steps, gamma=gamma_coef)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, verbose=True)

model.to(device)

best_val = 100

def create_small_class_mask( img, annot, coef = 64):
        height, width = img.shape[-2:]
        
        # target_map = np.zeros((int(height//coef), int(width//coef)))#, 2
        target_map = torch.zeros((img.shape[0], 1, int(height//coef), int(width//coef)))#, 2
        # target_map[:,:,0] = 1
        
        # if height % coef != 0 or width % coef != 0:
        #     pad_w = coef - height%coef
        #     pad_h = coef - width%coef
        #     new_image = np.zeros((height + pad_w, width + pad_h, cns)).astype(np.float32)
        #     new_image[:height, :width, :] = img.astype(np.float32)
        #     img = new_image
        #     height, width, cns = img.shape
        #     target_map = np.zeros((int(height//coef), int(width//coef)))#, 2
        
        for i in range(annot.shape[0]):
            for coord in annot[i]:
                if coord[4] == -1:
                    continue
                x1 = coord[0]
                y1 = coord[1]
                x2 = coord[2]
                y2 = coord[3]
                
                x = int(((x2 + x1) / 2) // coef)
                y = int(((y2 + y1) / 2) // coef)

                
                target_map[i,0,y, x] = 1
                
        
        return target_map


def train_one_epoch(epoch_num, train_data_loader):
    
    print("Train Epoch - {} Started".format(epoch_num), flush=True)
    st = time.time()
    
    model.train()
    
    epoch_loss = []

    for iter_num, data in enumerate(train_data_loader):
                
        optimizer.zero_grad()
        
        # img, annot = data['img'].cuda().float(), data['annot'].cuda().float()
        
        img = data[0]['data']
        bbox = data[0]['bboxe'].int()
        z = data[0]['img_id']
        h_flip = data[0]['horizontal_flip']
        v_flip = data[0]['vertical_flip']
        bb_shape = data[0]['bbox_shapes']
        bbox = flip_bboxes(bbox, h_flip, v_flip, bb_shape)
        placeholders = torch.full((bb_shape.shape[0], 60, 1), fill_value=0)
        annot = torch.cat((bbox, placeholders), dim=2)
        
        mask = create_small_class_mask(img, annot).to(device)
        
        pred = model(img)
        
        mask.to(device)
        loss1 = criterion1(pred, mask)
        loss2 = criterion2(pred, mask)
        loss = loss1 + loss2

        if bool(loss == 0):
            print(
            'Epoch: {} | Iteration: {} Map loss {:1.5f}| Running loss: {:1.5f}'.format(
                epoch_num, iter_num, float(loss) , np.mean(epoch_loss)), flush=True)
            continue
                
        loss.backward()


                
        optimizer.step()
        epoch_loss.append(float(loss))

            
        print(
            'Epoch: {} | Iteration: {} Map loss {:1.5f}| Running loss: {:1.5f}'.format(
                epoch_num, iter_num, float(loss) , np.mean(epoch_loss)), flush=True)
        
        
    # Update the learning rate
    if lr_scheduler is not None:
        lr_scheduler.step()
        
    et = time.time()
    print("\n Total Time - {}\n".format(int(et - st)))
    return np.mean(epoch_loss)
        
    
def get_metric_one_epoch(epoch_num, valid_data_loader, best_val):
    
    print("GetMetric Epoch - {} Started".format(epoch_num))
    st = time.time()
    model.eval()

    epoch_loss = []
    prediction = []
    
    accuracy_mean = []
    precision_mean = []
    recall_mean = []
    f1_mean = []
    iou = []
    dice = []

    f_coef = 0.5

    for iter_num, data in enumerate(valid_data_loader):
                
        with torch.no_grad():
            
            img = data[0]['data']
            bbox = data[0]['bboxe'].int()
            z = data[0]['img_id']
            bb_shape = data[0]['bbox_shapes'].cpu()
            
            placeholders = torch.full((bb_shape.shape[0], 60, 1), fill_value=0)
            annot = torch.cat((bbox, placeholders), dim=2)
            mask = create_small_class_mask(img, annot).to(device).long()
            
            pred1 = model(img)
            
            mask
            loss1 = criterion1(pred1, mask)
            loss2 = criterion2(pred1, mask)
            loss = loss1 + loss2
            
        epoch_loss.append(float(loss))
            
        print(
            'Epoch: {} | Iteration: {} Map loss {:1.5f}| Running loss: {:1.5f}'.format(
                epoch_num, iter_num, float(loss) , np.mean(epoch_loss)), flush=True)
        
        
            
        pred = (torch.sigmoid(pred1) > 0.5).cpu().long()
        mask = mask.cpu()
        
        # pred = pred.cpu()
        accuracy, precision, recall, f1 = calculate_semantic_metrics(pred, mask, f_coef = f_coef)
        
        accuracy_mean.append(accuracy)
        precision_mean.append(precision)
        recall_mean.append(recall)
        f1_mean.append(f1)
        
        iou.append(iou_pytorch(pred, mask))
        dice.append(dice_pytorch(pred, mask))
        
        
        
            
    print(f'Accuracy: {np.mean(accuracy_mean)}')
    print(f'Precision: {np.mean(precision_mean)}')
    print(f'Recall: {np.mean(recall_mean)}')
    print(f'F{f_coef} Score: {np.mean(f1_mean)}')
    print(f'iou: {np.mean(iou)}')
    print(f'dice: {np.mean(dice)}')
        

        
    et = time.time()
    print("\n Total Time - {}\n".format(int(et - st)))
    
    if best_val > np.mean(epoch_loss):
        best_val = np.mean(epoch_loss)
        torch.save(model, f"{path_to_save}_n:{epoch_num}_iou:{np.mean(iou)}_f:{np.mean(f1_mean)}_val:{best_val}.pt")
    elif epoch_num >= epochs - 1:
        torch.save(model, f"{path_to_save}_n:{epoch_num}_iou:{np.mean(iou)}_f:{np.mean(f1_mean)}_val:{np.mean(epoch_loss)}_last.pt")
    
    return np.mean(iou), np.mean(f1_mean), np.mean(epoch_loss), best_val
    
best_val = 100    

for epoch in range(epochs):
    st = time.time()

    mean_loss_train = train_one_epoch(epoch, dali_iterator_train)
    
    iou, f1_mean, mean_loss_val, best_val = get_metric_one_epoch(epoch, dali_iterator_val, best_val)
    et = time.time()
    
    wandb.log({"epoch": epoch, "train_loss": float(mean_loss_train), "val_loss": float(mean_loss_val), 
                   "iou": float(iou), "f1_mean": float(f1_mean), "total_time":int(et - st)})