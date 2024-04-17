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
import torch.nn as nn
import kornia 

from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

from utils.datasetLADD import LADD
from utils.dataloader import collater_annot, ToTorch, Augmenter, Normalizer, Resizer
from utils.Dali_resize import get_dali_pipeline, get_dali_pipeline_aug,get_dali_pipeline_small_aug, flip_bboxes2, flip_bboxes, resize_bb
from utils.metrics import evaluate
from utils.copypaste import CopyPasteAugmentation
from retinanet import model_oan
from retinanet import losses


# import torch.autograd
# torch.autograd.set_detect_anomaly(True)



epochs = 30
batch_size = 8
num_workers = 2
oan_gamma = 2
oan_alpha = 0.01
resize_to = (1024, 1024)
bb_pad = 0.5
model_name = f'retinanet_oan_resize_h:{resize_to[0]}_w:{resize_to[1]}'

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

print('ретина на more summer предобученная')

weights_name = f'{datetime.date.today().isoformat()}_{model_name}_vis+small+main_lr:{start_lr}_step:{num_steps}_gamma:{oan_gamma}_alpha:{oan_alpha}'
# weights_name = f'{datetime.date.today().isoformat()}_retinanet_oan_vis+small_lr_lr{start_lr}_step{num_steps}_gamma{oan_gamma}_alpha{oan_alpha}'
path_to_save = f'/home/maantonov_1/VKR/weights/retinanet/resize/main/{datetime.date.today().isoformat()}/gamma{oan_gamma}_alpha{oan_alpha}/'
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)
path_to_save = path_to_save + weights_name

print(f'path_to_save {path_to_save}')


# path_to_weights = '/home/maantonov_1/VKR/weights/retinanet/visdrone/retinanet_oan_vis_lr0.0003_step_5_gamma:2_alpha:0.25_n26_m:0.03_f:0.04.pt'
path_to_weights = '/home/maantonov_1/VKR/weights/retinanet/resize/small/2024-03-23/gamma2_alpha0.1/2024-03-23_retinanet_oan_resize_h:1024_w:1024_vis+small_lr:0.0003_step:10_gamma:2_alpha:0.1_n28_m:0.45_f:0.24_val:0.5285.pt'

os.environ["WANDB_MODE"]="offline" 
wandb.init(project="VKR", entity="matvey_antonov", name = f"{weights_name}")


# main_dir = '/home/maantonov_1/VKR/data/crope_data/main/crop_train_1024/'
# main_dir = '/home/maantonov_1/VKR/data/crope_data/small/small_crop_train/'
main_dir = '/home/maantonov_1/VKR/data/main_data/train/'
# main_dir = '/home/maantonov_1/VKR/data/small_train/train/'
images_dir_tarin = main_dir + 'images'
# annotations_file_train = main_dir + 'train_annot/annot.json'
annotations_file_train = main_dir + 'more_sum_train_annot/annot.json'

# main_dir = '/home/maantonov_1/VKR/data/crope_data/main/crop_val_1024/'
# main_dir = '/home/maantonov_1/VKR/data/crope_data/small/small_crop_val/'
main_dir = '/home/maantonov_1/VKR/data/main_data/train/'
# main_dir = '/home/maantonov_1/VKR/data/small_train/train/'
images_dir_val = main_dir + 'images'
annotations_file_val = main_dir + 'val_annot/annot.json'

main_dir = '/home/maantonov_1/VKR/data/main_data/test/'
images_dir_test = main_dir + 'images'
annotations_file_test = main_dir + 'annotations/annot.json'


dali_iterator_train = DALIGenericIterator(
    pipelines=[get_dali_pipeline_small_aug(images_dir = images_dir_tarin, annotations_file = annotations_file_train, resize_dims = resize_to, batch_size = batch_size, num_threads = num_workers)],
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

dali_iterator_test = DALIGenericIterator(
    pipelines=[get_dali_pipeline(images_dir = images_dir_test, annotations_file = annotations_file_test, resize_dims = resize_to, batch_size = batch_size, num_threads = num_workers)],
    output_map=['data', 'bboxe', 'bbox_shapes', 'img_id', 'original_sizes'],
    reader_name='Reader',
    last_batch_policy=LastBatchPolicy.PARTIAL,
    auto_reset=True,
    dynamic_shape=True
)

print(f'dataset Created', flush=True)


transform = nn.Sequential(
    kornia.augmentation.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue = 0.01, p=0.3),
    kornia.augmentation.RandomGaussianNoise(mean=0., std=0.05, p=.1),
    kornia.augmentation.RandomMedianBlur(kernel_size=(3, 7), p=0.2),
    kornia.augmentation.RandomMotionBlur(kernel_size=(3, 7),angle=25., direction=(-1,1), p = 0.2),
    kornia.augmentation.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
)


path = ['/home/maantonov_1/VKR/data/copypaste/resize_crop_main']#,'/home/maantonov_1/VKR/data/copypaste/resize_crop_small']
cp = CopyPasteAugmentation(path, max_objects = 3, random_state = None, feather_amount = 31, bb_pad = bb_pad)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()
print('CUDA available: {}'.format(torch.cuda.is_available()), flush=True)
print(f'Crreating retinanet ===>', flush=True)

# retinanet = model_oan.resnet50(num_classes = 2, pretrained = False, inputs = 3)


retinanet = torch.load(path_to_weights, map_location=device)

retinanet.focalLoss = losses.FocalLoss(alpha = oan_alpha, gamma = oan_gamma)
retinanet.oan_loss = losses.OANFocalLoss(alpha = oan_alpha, gamma = oan_gamma)


optimizer = optim.AdamW(retinanet.parameters(), lr = start_lr)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = num_steps, gamma=gamma_coef)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, verbose=True)

retinanet.to(device)



def train_one_epoch(epoch_num, train_data_loader):
    torch.cuda.empty_cache()
    print("Train Epoch - {} Started".format(epoch_num), flush=True)
    st = time.time()
    
    retinanet.train()
    
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
        
        bbox = resize_bb(bbox, original_sizes, bb_pad = bb_pad, new_shape = resize_to)
        bbox = flip_bboxes(bbox, h_flip, v_flip, bb_shape,img_size=resize_to)
        
        new_elements = torch.where(bbox[:, :, 3] == -1, -1, 1).unsqueeze(2)  
        annot = torch.cat((bbox, new_elements), dim=2).to(device)
        
        img, annot = cp.apply_augmentation(img, annot)
        img = transform(img)
        
        classification_loss, regression_loss, oan_loss, _ = retinanet([img, annot])
                
        classification_loss = classification_loss.mean()
        regression_loss = regression_loss.mean()

        loss = classification_loss + regression_loss + 4 * oan_loss
        

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
    torch.cuda.empty_cache()
    print("Val Epoch - {} Started".format(epoch_num))
    st = time.time()
    retinanet.train()

    epoch_loss = []

    for iter_num, data in enumerate(valid_data_loader):
                
        with torch.no_grad():
            
            img = data[0]['data']
            bbox = data[0]['bboxe'].int()
            z = data[0]['img_id']
            bb_shape = data[0]['bbox_shapes']
            original_sizes = data[0]['original_sizes']
            bbox = resize_bb(bbox, original_sizes, bb_pad = bb_pad, new_shape = resize_to)
        
            new_elements = torch.where(bbox[:, :, 3] == -1, -1, 1).unsqueeze(2)  
            annot = torch.cat((bbox, new_elements), dim=2).to(device)
            
            classification_loss, regression_loss, oan_loss, _ = retinanet([img, annot])
                    
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            

            loss = classification_loss + regression_loss + 4 * oan_loss

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
    
def get_metric_one_epoch(epoch_num, valid_data_loader, best_val, last_val, mode = 'val'):
    
    torch.cuda.empty_cache()
    
    print("GetMetric Epoch - {} Started".format(epoch_num))
    st = time.time()
    retinanet.eval()

    epoch_loss = []
    prediction = []
    time_running = []

    for iter_num, data in enumerate(valid_data_loader):
                
        with torch.no_grad():
            
            img = data[0]['data']
            bbox = data[0]['bboxe']
            z = data[0]['img_id']
            bb_shape = data[0]['bbox_shapes'].cpu()
            original_sizes = data[0]['original_sizes']
            bbox = resize_bb(bbox, original_sizes, bb_pad = bb_pad, new_shape = resize_to).int()
            
            new_elements = torch.where(bbox[:, :, 3] == -1, -1, 1).unsqueeze(2)  
            annot = torch.cat((bbox, new_elements), dim=2).cpu()
            
            t = time.time()
            scores, labels, boxes = retinanet(img)
            t1 = time.time()
            
            pred_dict = {}
            
        time_running.append(t1-t)
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy() * 0
        boxes  = boxes.cpu().numpy() 
        pred_dict['scores'] = scores
        pred_dict['labels'] = labels
        pred_dict['boxes']  = boxes
        
        
        active_annot = [annot[i, :bb_shape[i, 0]] for i in range(bb_shape.size(0))]
        active_annot_tensor = torch.cat(active_annot, dim=0)
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
        
    if mode == 'test':
        print(f'{datetime.date.today().isoformat()}')
        print(f'path_to_save {path_to_save}')
        print(f'AVG time: {np.mean(time_running)}')
        print()

        for i_tresh in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            score_threshold = i_tresh
            map_score, Fscore = evaluate(prediction, score_threshold = score_threshold)
            print(f'score_threshold: {score_threshold}')
            print(f'map_score: {map_score}')
            print(f'Fscore: {Fscore}')
            print()
        return
        
 
    map_score, Fscore = evaluate(prediction)
            
    print(
            'Epoch: {} | map_score loss: {:1.5f} | Fscore loss: {:1.5f} | AVG time: {:1.5f}'.format(
                epoch_num, float(map_score), float(Fscore), float(np.mean(time_running))), flush=True)
        


    et = time.time()
    print("\n Total Time - {}\n".format(int(et - st)))
    
    # if best_val > last_val:
    #     best_val = last_val
    #     torch.save(retinanet, f"{path_to_save}_n{epoch_num}_m:{map_score:0.2f}_f:{Fscore:0.2f}_val:{best_val:0.4f}.pt")
    # elif epoch_num >= epochs - 1:
    #     torch.save(retinanet, f"{path_to_save}_n{epoch_num}_m:{map_score:0.2f}_f:{Fscore:0.2f}_val:{last_val:0.4f}_last.pt")
        
    return map_score, Fscore, best_val
    
best_val = 100
    
for epoch in range(epochs):
    st = time.time()

    mean_loss_train = train_one_epoch(epoch, dali_iterator_train)
    
    mean_loss_val = valid_one_epoch(epoch, dali_iterator_val)

    map_score, Fscore, best_val = get_metric_one_epoch(epoch, dali_iterator_val, best_val, mean_loss_val)
    et = time.time()
    
    wandb.log({"epoch": epoch, "train_loss": float(mean_loss_train), "val_loss": float(mean_loss_val), 
                   "map_score": float(map_score), "Fscore": float(Fscore), "total_time":int(et - st)})
    
    
get_metric_one_epoch(0, dali_iterator_test, 0, 0, mode = 'test')