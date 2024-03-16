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
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms as T
import torchvision.transforms.functional as F
import albumentations as A
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

from utils.datasetLADD import LADD
from utils.dataloader import collater_annot, ToTorch, Augmenter, Normalizer, Resizer
from utils.Dali import get_dali_pipeline, get_dali_pipeline_aug, flip_bboxes2
from utils.metrics import evaluate



# import torch.autograd
# torch.autograd.set_detect_anomaly(True)
def convert_annotations(image, annotations):
    batch_size, _, _ = annotations.shape
    targets = []
    images = []
    for i in range(batch_size):
        images.append(image[i])
        boxes = []
        labels = []
        for j in range(60):  # 60 - максимальное предполагаемое количество аннотаций на изображение
            if annotations[i, j, 3] != -1:  # Проверяем, не является ли аннотация заполнителем
                boxes.append(annotations[i, j, :4])
                labels.append(int(annotations[i, j, 4]))
        
        if len(boxes) != 0:
            boxes = torch.stack(boxes)
            targets.append({'boxes': boxes, 'labels': torch.tensor(labels)})
        else:
            targets.append({'boxes': torch.tensor([]), 'labels': torch.tensor([])})
    return images, targets


epochs = 15
batch_size = 14
num_workers = 4
oan_gamma = 2
oan_alpha = 0.10

#optimazer
start_lr   = 0.0003
num_steps  = 5
gamma_coef = 0.5

weights_name = f'retinanet_oan_vis+small_lr+ful_lr{start_lr}_step{num_steps}_gamma{oan_gamma}_alpha{oan_alpha}'
path_to_save = '/home/maantonov_1/VKR/weights/retinanet/10_03_2024/' + weights_name

path_to_weights = '/home/maantonov_1/VKR/weights/retinanet/05_03_2024/retinanet_oan_vis+small_lr+ful_0.0003_step5_0_0.5934904131544642.pt'

os.environ["WANDB_MODE"]="offline" 
wandb.init(project="VKR", entity="matvey_antonov", name = f"{weights_name}_{datetime.date.today().isoformat()}")


main_dir = '/home/maantonov_1/VKR/data/crope_data/main/crop_train_1024/'
images_dir_tarin = main_dir + 'images'
annotations_file_train = main_dir + 'annotations/annot.json'

main_dir = '/home/maantonov_1/VKR/data/crope_data/main/crop_val_1024/'
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
print(f'Crreating retinanet ===>', flush=True)


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

num_classes = 2
model = get_model(num_classes)


optimizer = optim.Adam(model.parameters(), lr = start_lr)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = num_steps, gamma=gamma_coef)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, verbose=True)

model = model.to(device)

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
        bbox = flip_bboxes2(bbox, h_flip, v_flip, bb_shape)
        placeholders = torch.full((bb_shape.shape[0], 60, 1), fill_value=0)
        annot = torch.cat((bbox, placeholders), dim=2).to(device)
        
        images, targets = convert_annotations(img, annot)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
                
        loss = sum(v for v in loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        if bool(loss == 0):
            print(
            'Epoch: {} | Iteration: {} | loss loss: {:1.5f}'.format(
                epoch_num, iter_num, float(loss)), flush=True)
            continue
                
        
        epoch_loss.append(float(loss))

            
        print(
            'Epoch: {} | Iteration: {} | loss loss: {:1.5f}'.format(
                epoch_num, iter_num, float(loss)), flush=True)
        
        
    # Update the learning rate
    if lr_scheduler is not None:
        lr_scheduler.step()
        
    et = time.time()
    print("\n Total Time - {}\n".format(int(et - st)))
    return np.mean(epoch_loss)
        
def valid_one_epoch(epoch_num, valid_data_loader):
    
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
            placeholders = torch.full((bb_shape.shape[0], 60, 1), fill_value=0)
            annot = torch.cat((bbox, placeholders), dim=2).to(device)
            
            images, targets = convert_annotations(image, annot)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
                    
            loss = sum(v for v in loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.append(float(loss))

            print(
            'Epoch: {} | Iteration: {} | loss loss: {:1.5f}'.format(
                epoch_num, iter_num, float(loss)), flush=True)

        
    et = time.time()
    print("\n Total Time - {}\n".format(int(et - st)))
    
    return np.mean(epoch_loss)
    
def get_metric_one_epoch(epoch_num, valid_data_loader):
    
    print("GetMetric Epoch - {} Started".format(epoch_num))
    st = time.time()
    retinanet.eval()

    epoch_loss = []
    prediction = []

    for iter_num, data in enumerate(valid_data_loader):
                
        with torch.no_grad():
            
            img = data[0]['data']
            bbox = data[0]['bboxe'].int()
            z = data[0]['img_id']
            bb_shape = data[0]['bbox_shapes'].cpu()
            
            placeholders = torch.full((bb_shape.shape[0], 60, 1), fill_value=0)
            annot = torch.cat((bbox, placeholders), dim=2).cpu()
            
            scores, labels, boxes = retinanet(img)
            
            pred_dict = {}
            
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        boxes  = boxes.cpu().numpy()
        pred_dict['scores'] = scores
        pred_dict['labels'] = labels
        pred_dict['boxes']  = boxes
        
        
        active_annot = [annot[i, :bb_shape[i, 0]] for i in range(bb_shape.size(0))]
        active_annot_tensor = torch.cat(active_annot, dim=0)
        # active_annot_np = active_annot_tensor.numpy()
        
        gt_dict = {}
        if annot.shape[0] != 0:
            gt_labels = active_annot_tensor[:,-1].numpy()
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
    

    torch.save(retinanet, f"{path_to_save}_{epoch_num}_{map_score}_{Fscore}.pt")
    return map_score, Fscore
    
    
for epoch in range(epochs):
    st = time.time()

    mean_loss_train = train_one_epoch(epoch, dali_iterator_train)
    
    mean_loss_val = valid_one_epoch(epoch, dali_iterator_val)
    
    map_score, Fscore = get_metric_one_epoch(epoch, dali_iterator_val)
    et = time.time()
    
    wandb.log({"epoch": epoch, "train_loss": float(mean_loss_train), "val_loss": float(mean_loss_val), 
                   "map_score": float(map_score), "Fscore": float(Fscore), "total_time":int(et - st)})