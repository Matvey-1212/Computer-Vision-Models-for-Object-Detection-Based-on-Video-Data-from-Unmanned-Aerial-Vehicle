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
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms as T
import torchvision.transforms.functional as F
import albumentations as A
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import torchvision
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights

from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

from utils.datasetLADD import LADD
from utils.dataloader import collater_annot, ToTorch, Augmenter, Normalizer, Resizer
from utils.Dali import get_dali_pipeline, get_dali_pipeline_aug, flip_bboxes2, flip_bboxes
from utils.metrics import evaluate

import sys
sys.path.append('./detr/')

from detr.models.matcher import HungarianMatcher
from detr.models.detr import SetCriterion



# import torch.autograd
# torch.autograd.set_detect_anomaly(True)



epochs = 50
batch_size = 20
num_workers = 2
null_class_coef = 0.1
num_queries = 100
loss_ce_w = 2
loss_bbox_w = 1
loss_giou_w = 1


#optimazer
start_lr   = 0.0003
num_steps  = 15
gamma_coef = 0.5

print(f'epochs {epochs}')
print(f'batch_size {batch_size}')
print(f'num_workers {num_workers}')

print(f'null_class_coef {null_class_coef}')
print(f'num_queries {num_queries}')
print(f'loss_ce_w {loss_ce_w}')
print(f'loss_bbox_w {loss_bbox_w}')
print(f'loss_giou_w {loss_giou_w}')

print(f'start_lr {start_lr}')
print(f'num_steps {num_steps}')
print(f'gamma_coef {gamma_coef}')

weights_name = f'{datetime.date.today().isoformat()}_detr_main_lr:{start_lr}_step:{num_steps}'
path_to_save = f'/home/maantonov_1/VKR/weights/detr/{datetime.date.today().isoformat()}/{int(time.time())}'
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



def transform_annotations(batch_imgs, batch_annots, img_id):
    transformed_annotations = []
    transformed_img = []
    _,_, h,w = batch_imgs.shape
    for i in range(batch_annots.shape[0]): 
        annots = batch_annots[i].to(device)
        valid_annots = annots[annots[:, 3] != -1]  
        transformed_img.append(batch_imgs[i])
        
        
        valid_annots[:, 2] = (valid_annots[:, 2] - valid_annots[:, 0]) / w
        valid_annots[:, 3] = (valid_annots[:, 3] - valid_annots[:, 1]) / h
        valid_annots[:, 0] = valid_annots[:, 0] / w
        valid_annots[:, 1] = valid_annots[:, 1] / h
        
        img_annots = {
            'boxes': valid_annots[:, :4].to(device).float(),  
            'labels': torch.tensor([0] * valid_annots.shape[0]).to(device).long(),  
            'area': valid_annots[:, 2] * valid_annots[:, 3],
            'image_id': img_id[i]
        }
        transformed_annotations.append(img_annots)
    return transformed_img, transformed_annotations

def to_numpy(dic):
    for key in dic:
        dic[key] = dic[key].cpu().numpy()
        
       
class DETRModel(nn.Module):
    def __init__(self,num_classes,num_queries):
        super(DETRModel,self).__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        
        self.model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
        self.in_features = self.model.class_embed.in_features
        
        self.model.class_embed = nn.Linear(in_features=self.in_features,out_features=self.num_classes)
        self.model.num_queries = self.num_queries
        
    def forward(self,images):
        return self.model(images) 



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()
print('CUDA available: {}'.format(torch.cuda.is_available()), flush=True)
print(f'Crreating model ===>', flush=True)

matcher = HungarianMatcher()
weight_dict = weight_dict = {'loss_ce': loss_ce_w, 'loss_bbox': loss_bbox_w , 'loss_giou': loss_giou_w}
losses = ['labels', 'boxes', 'cardinality']

model = DETRModel(num_classes=2,num_queries=num_queries)

model = model.to(device)
criterion = SetCriterion(1, matcher, weight_dict, eos_coef = null_class_coef, losses=losses)
criterion = criterion.to(device)



optimizer = optim.AdamW(model.parameters(), lr = start_lr)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = num_steps, gamma=gamma_coef)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, verbose=True)

model.to(device)

# def dict_list_to_dict(list_of_dicts):
#     concatenated_dict = {}
#     for key in list_of_dicts[0]:
#         concatenated_dict[key] = torch.cat([d[key] for d in list_of_dicts], dim=0)
#     return concatenated_dict

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
                
        try:
            optimizer.zero_grad()
            
            # img, annot = data['img'].cuda().float(), data['annot'].cuda().float()
            
            img = data[0]['data']
            bbox = data[0]['bboxe'].int()
            img_id = data[0]['img_id']
            h_flip = data[0]['horizontal_flip']
            v_flip = data[0]['vertical_flip']
            bb_shape = data[0]['bbox_shapes']
            bbox = flip_bboxes(bbox, h_flip, v_flip, bb_shape)

            
            img, annot = transform_annotations(img, bbox, img_id)
            
            output = model(img)
            
            
            
            loss_dict = criterion(output, annot)
                
            weight_dict = criterion.weight_dict
        
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            

            if bool(losses == 0):
                print(
                'Epoch: {} | Iteration: {} |  loss: {:1.5f} | Running loss: {:1.5f}'.format(
                    epoch_num, iter_num, float(losses), np.mean(epoch_loss)), flush=True)
                continue
                    
            losses.backward()

                    
            optimizer.step()
            epoch_loss.append(float(losses))

            
            print(
                'Epoch: {} | Iteration: {} |  loss: {:1.5f} | Running loss: {:1.5f}'.format(
                    epoch_num, iter_num, float(losses), np.mean(epoch_loss)), flush=True)
        except:
            print(f'Error!')
        
        
        
    # Update the learning rate
    if lr_scheduler is not None:
        lr_scheduler.step()
        
    et = time.time()
    print("\n Total Time - {}\n".format(int(et - st)))
    return np.mean(epoch_loss)
        
def valid_one_epoch(epoch_num, valid_data_loader):
    
    print("Val Epoch - {} Started".format(epoch_num))
    st = time.time()
    model.train()

    epoch_loss = []

    for iter_num, data in enumerate(valid_data_loader):
                
        try:
            with torch.no_grad():
                
                img = data[0]['data']
                bbox = data[0]['bboxe'].int()
                img_id = data[0]['img_id']
                
                img, annot = transform_annotations(img, bbox, img_id)
            
                output = model(img)
                
                
                
                loss_dict = criterion(output, annot)
                weight_dict = criterion.weight_dict
            
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

                epoch_loss.append(float(losses))

                print(
                'Epoch: {} | Iteration: {} |  loss: {:1.5f} | Running loss: {:1.5f}'.format(
                    epoch_num, iter_num, float(losses), np.mean(epoch_loss)), flush=True)
        except:
            print(f'Error!')
        
        
        
    last_val = np.mean(epoch_loss)
    
    et = time.time()
    print("\n Total Time - {}\n".format(int(et - st)))
    
    return np.mean(epoch_loss), last_val
    
def get_metric_one_epoch(epoch_num, valid_data_loader, last_val):
    
    print("GetMetric Epoch - {} Started".format(epoch_num))
    st = time.time()
    model.eval()

    epoch_loss = []
    prediction = []

    for iter_num, data in enumerate(valid_data_loader):
        try:
                
            with torch.no_grad():
                
                img = data[0]['data']
                bbox = data[0]['bboxe'].int()
                img_id = data[0]['img_id']
                bb_shape = data[0]['bbox_shapes'].cpu()
                _,_,h,w = img.shape
                

                
                img, annot = transform_annotations(img, bbox, img_id)
                outputs = model(img)
                
                outputs = [{k: v.cpu() for k, v in outputs.items()}]
                
                
                pred_list = []
                annot_list = []
                for i in range(len(outputs)):
                    
                    local_dict = {}
                    boxes = outputs[0]['pred_boxes'][i].detach().cpu().numpy()
                    prob   = outputs[0]['pred_logits'][i].softmax(1).detach().cpu().numpy()[:,0]
                    
                    boxes[:, 0] = boxes[:, 0] * w
                    boxes[:, 1] = boxes[:, 1] * h
                    boxes[:, 2] = boxes[:, 0] + boxes[:, 2] * w
                    boxes[:, 3] = boxes[:, 1] + boxes[:, 3] * h
                    
                    local_dict['boxes'] = boxes
                    local_dict['scores'] = prob
                    local_dict['labels'] = np.zeros(prob.shape[0])
                    
                    pred_list.append(local_dict)
                    
                    local_annot = {}
                    local_annot['boxes'] = bbox[i].cpu().numpy()
                    local_annot['labels'] = np.zeros(bbox[i].shape[0])
                    annot_list.append(local_annot)
                    
                    
                
            
            annot, pred_dict = dict_list_to_dict(annot_list), dict_list_to_dict(pred_list)
                
                
            # to_numpy(annot)
            # to_numpy(pred_dict)

            prediction.append((annot, pred_dict))
        except:
            print(f'Error!')
        
        
    map_score, Fscore = evaluate(prediction)
            
    print(
            'Epoch: {} | map_score loss: {:1.5f} | Fscore loss: {:1.5f} '.format(
                epoch_num, float(map_score), float(Fscore)), flush=True)
        

        
    et = time.time()
    print("\n Total Time - {}\n".format(int(et - st)))
    
    # if best_val > last_val:
    #     best_val = last_val
    #     torch.save(model, f"{path_to_save}_n{epoch_num}_m:{map_score:0.2f}_f:{Fscore:0.2f}_val:{best_val:0.4f}.pt")
    # elif epoch_num >= epochs - 1:
    #     torch.save(model, f"{path_to_save}_n{epoch_num}_m:{map_score:0.2f}_f:{Fscore:0.2f}_val:{last_val:0.4f}_last.pt")
        
    if epoch_num%5 == 0:
        torch.save(model, f"{path_to_save}_n{epoch_num}_m:{map_score:0.2f}_f:{Fscore:0.2f}_val:{last_val:0.4f}.pt")
    elif epoch_num >= epochs - 1:
        torch.save(model, f"{path_to_save}_n{epoch_num}_m:{map_score:0.2f}_f:{Fscore:0.2f}_val:{last_val:0.4f}_last.pt")
        
    return map_score, Fscore
    
best_val = 100
last_val = 0
    
for epoch in range(epochs):
    st = time.time()

    mean_loss_train = train_one_epoch(epoch, dali_iterator_train)
    
    mean_loss_val, last_val = valid_one_epoch(epoch, dali_iterator_val)
    
    map_score, Fscore = get_metric_one_epoch(epoch, dali_iterator_val, last_val)
    et = time.time()
    
    wandb.log({"epoch": epoch, "train_loss": float(mean_loss_train), "val_loss": float(mean_loss_val), 
                   "map_score": float(map_score), "Fscore": float(Fscore), "total_time":int(et - st)})