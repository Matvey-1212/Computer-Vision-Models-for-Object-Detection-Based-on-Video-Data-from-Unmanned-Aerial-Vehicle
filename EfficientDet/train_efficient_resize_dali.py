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

# from utils.datasetLADD import LADD
# from utils.dataloader import collater_annot, ToTorch, Augmenter, Normalizer, Resizer
from UTILS.Dali_resize import get_dali_pipeline, get_dali_pipeline_aug, flip_bboxes2, flip_bboxes, resize_bb
from UTILS.metrics import evaluate
from backbone import EfficientDetBackbone
from efficientdet.loss import FocalLoss
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import  postprocess

# import torch.autograd
# torch.autograd.set_detect_anomaly(True)



epochs = 30
batch_size = 2
num_workers = 2
oan_gamma = 2
oan_alpha = 0.25
resize_to = (1024, 1024)
model_name = f'efficientb4_resize_h:{resize_to[0]}_w:{resize_to[1]}'


#optimazer
start_lr   = 0.0003
num_steps  = 10
gamma_coef = 0.5

print(f'epochs {epochs}')
print(f'batch_size {batch_size}')
print(f'num_workers {num_workers}')
print(f'oan_gamma {oan_gamma}')
print(f'oan_alpha {oan_alpha}')
print(f'start_lr {start_lr}')
print(f'num_steps {num_steps}')
print(f'gamma_coef {gamma_coef}')
print(f'resize_to {resize_to}')

weights_name = f'{datetime.date.today().isoformat()}_{model_name}_small+main_lr:{start_lr}_step:{num_steps}_gamma:{oan_gamma}_alpha:{oan_alpha}'

path_to_save = f'/home/maantonov_1/VKR/weights/efficient/resize/main/{datetime.date.today().isoformat()}/gamma{oan_gamma}_alpha{oan_alpha}/'
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)
path_to_save = path_to_save + weights_name

print(f'path_to_save {path_to_save}')


path_to_weights = '/home/maantonov_1/VKR/weights/efficient/resize/small/2024-03-24/gamma2_alpha0.5/2024-03-24_efficientb4_resize_h:1024_w:1024_small_lr:0.0003_step:10_gamma:2_alpha:0.5_n29_m:0.42_f:0.12_val:2.1582.pt'

os.environ["WANDB_MODE"]="offline" 
wandb.init(project="VKR", entity="matvey_antonov", name = f"{weights_name}")


# main_dir = '/home/maantonov_1/VKR/data/crope_data/main/crop_train_1024/'
# main_dir = '/home/maantonov_1/VKR/data/crope_data/small/small_crop_train/'
main_dir = '/home/maantonov_1/VKR/data/main_data/train/'
# main_dir = '/home/maantonov_1/VKR/data/small_train/train/'
images_dir_tarin = main_dir + 'images'
annotations_file_train = main_dir + 'train_annot/annot.json'

# main_dir = '/home/maantonov_1/VKR/data/crope_data/main/crop_val_1024/'
# main_dir = '/home/maantonov_1/VKR/data/crope_data/small/small_crop_val/'
main_dir = '/home/maantonov_1/VKR/data/main_data/train/'
# main_dir = '/home/maantonov_1/VKR/data/small_train/train/'
images_dir_val = main_dir + 'images'
annotations_file_val = main_dir + 'val_annot/annot.json'


dali_iterator_train = DALIGenericIterator(
    pipelines=[get_dali_pipeline_aug(images_dir = images_dir_tarin, annotations_file = annotations_file_train, resize_dims = resize_to, batch_size = batch_size, num_threads = num_workers)],
    output_map=['data', 'bboxe', 'bbox_shapes', 'img_id', 'horizontal_flip','vertical_flip', 'original_sizes'],
    reader_name='Reader',
    last_batch_policy=LastBatchPolicy.PARTIAL,
    auto_reset=True,
    dynamic_shape=True
)

dali_iterator_val = DALIGenericIterator(
    pipelines=[get_dali_pipeline(images_dir = images_dir_val, annotations_file = annotations_file_val,resize_dims = resize_to, batch_size = batch_size, num_threads = num_workers)],
    output_map=['data', 'bboxe', 'bbox_shapes', 'img_id', 'original_sizes'],
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


anchors_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
anchors_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]

# model = EfficientDetBackbone(num_classes=2, compound_coef=4,
#                                  ratios=anchors_ratios, scales=anchors_scales)

model = torch.load(path_to_weights, map_location=device)
criterion = FocalLoss(gamma = oan_gamma, alpha = oan_alpha)

optimizer = optim.Adam(model.parameters(), lr = start_lr)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = num_steps, gamma=gamma_coef)


model.to(device)

def dict_list_to_dict(list_of_dicts):
    concatenated_dict = {}

    for key in list_of_dicts[0]:
        temp_list = []
        for val in list_of_dicts:
            if val[key].shape[0] != 0:
                temp_list.append(val[key])
        
        if len(temp_list) > 0:
            
            concatenated_dict[key] = np.concatenate(temp_list, axis=0)
    
    if len(concatenated_dict) != 3:
        return {'boxes':np.array([]),'labels':np.array([]),'scores':np.array([])}
     
    return concatenated_dict



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
        original_sizes = data[0]['original_sizes']
        
        bbox = resize_bb(bbox, original_sizes, bb_pad = 0.1, new_shape = resize_to)
        bbox = flip_bboxes(bbox, h_flip, v_flip, bb_shape)
        new_elements = torch.where(bbox[:, :, 3] == -1, -1, 1).unsqueeze(2)  
        annot = torch.cat((bbox, new_elements), dim=2).to(device)
        
        
        features, regression, classification, anchors = model(img)

        cls_loss, reg_loss = criterion(classification, regression, anchors, annot)

        cls_loss = cls_loss.mean()
        reg_loss = reg_loss.mean()
                
        loss = cls_loss + reg_loss
        if loss == 0 or not torch.isfinite(loss):
            print(f'Zero Loss')
            continue
                
        loss.backward()
                
        optimizer.step()
        epoch_loss.append(float(loss))

            
        print(
            'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f}  | Running loss: {:1.5f}'.format(
                epoch_num, iter_num, float(cls_loss), float(reg_loss),  np.mean(epoch_loss)), flush=True)


        
    # Update the learning rate
    if lr_scheduler is not None:
        lr_scheduler.step()
        
    et = time.time()
    print("\n Total Time - {}\n".format(int(et - st)))
    return np.mean(epoch_loss)


        
def valid_one_epoch(epoch_num, valid_data_loader, best_val):
    
    print("Val Epoch - {} Started".format(epoch_num))
    st = time.time()
    model.train()

    epoch_loss = []
    prediction = []
    
    for iter_num, data in enumerate(valid_data_loader):
                
        with torch.no_grad():
            
            img = data[0]['data']
            bbox = data[0]['bboxe'].int()
            z = data[0]['img_id']
            bb_shape = data[0]['bbox_shapes']
            original_sizes = data[0]['original_sizes']
        
            bbox = resize_bb(bbox, original_sizes, bb_pad = 0.1, new_shape = resize_to)
            new_elements = torch.where(bbox[:, :, 3] == -1, -1, 1).unsqueeze(2)  
            annot = torch.cat((bbox, new_elements), dim=2).to(device)
            
            t = time.time()
            features, regression, classification, anchors = model(img)
            t1 = time.time()
            cls_loss, reg_loss = criterion(classification, regression, anchors, annot)
            t2 = time.time()
            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()
            pred_dict = postprocess(img,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            threshold = 0.05, iou_threshold = 0.2)
            
            
            t3 = time.time()

            cls_loss = cls_loss.mean()
            reg_loss = reg_loss.mean()
            loss = cls_loss + reg_loss
            epoch_loss.append(float(loss))

            print(
            'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f}  | Running loss: {:1.5f}'.format(
                epoch_num, iter_num, float(cls_loss), float(reg_loss),  np.mean(epoch_loss)), end = ' ', flush=True)
            
            print(f'predict: {t1 - t}ms, nms: {t3 - t2}')
            
        
        pred_dict = dict_list_to_dict(pred_dict)
            
        pred_dict['labels'] = pred_dict['labels'] - 1
            
        active_annot = [annot[i, :bb_shape[i, 0]] for i in range(bb_shape.size(0))]
        active_annot_tensor = torch.cat(active_annot, dim=0).cpu()
        # active_annot_np = active_annot_tensor.numpy()
        
        gt_dict = {}
        if annot.shape[0] != 0:
            gt_labels = active_annot_tensor[:,-1].numpy() * 0 
            gt_boxes  = active_annot_tensor[:,:-1].numpy()
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
    
    if best_val > np.mean(epoch_loss):
        best_val = np.mean(epoch_loss)
        torch.save(model, f"{path_to_save}_n{epoch_num}_m:{map_score:0.2f}_f:{Fscore:0.2f}_val:{best_val:0.4f}.pt")
    elif epoch_num >= epochs - 1:
        torch.save(model, f"{path_to_save}_n{epoch_num}_m:{map_score:0.2f}_f:{Fscore:0.2f}_val:{np.mean(epoch_loss):0.4f}_last.pt")
        
    return map_score, Fscore, np.mean(epoch_loss), best_val
    
    
    
best_val = 100
map_score, Fscore, mean_loss_val = 0, 0, 100
    
for epoch in range(epochs):
    st = time.time()

    mean_loss_train = train_one_epoch(epoch, dali_iterator_train)
    
    if epoch > 1:
        map_score, Fscore, mean_loss_val, best_val = valid_one_epoch(epoch, dali_iterator_val, best_val)

    et = time.time()
    
    wandb.log({"epoch": epoch, "train_loss": float(mean_loss_train), "val_loss": float(mean_loss_val), 
                   "map_score": float(map_score), "Fscore": float(Fscore), "total_time":int(et - st)})