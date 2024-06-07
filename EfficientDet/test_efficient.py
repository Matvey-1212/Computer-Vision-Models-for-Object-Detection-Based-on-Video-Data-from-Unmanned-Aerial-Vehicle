import os
import time
import numpy as np
import pandas as pd
import random
import datetime


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

from UTILS.datasetLADD import LADD
from UTILS.dataloader import collater_annot, ToTorch, Augmenter, Normalizer, Resizer
from UTILS.Dali import get_dali_pipeline, get_dali_pipeline_aug, flip_bboxes2, flip_bboxes
from UTILS.metrics import evaluate
from backbone import EfficientDetBackbone
from efficientdet.loss import FocalLoss
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import  postprocess

print('Started')


test_df = [{'dataframe': pd.read_csv('/home/maantonov_1/VKR/data/main_data/test/test_main.csv'),
             'image_dir': '/home/maantonov_1/VKR/data/main_data/test/images'}]

score_threshold = 0.05

# test_df = [{'dataframe': pd.read_csv('/home/maantonov_1/VKR/data/crope_data/main/crop_val_1024/crop_val_1024.csv'),
#              'image_dir': '/home/maantonov_1/VKR/data/crope_data/main/crop_val_1024/images'}]


test_dataset = LADD(test_df, mode = "valid", smart_crop = True, new_shape = (1024,1024), transforms = T.Compose([Normalizer(), ToTorch()]))#, Resizer()]))

print(f'dataset Created', flush=True)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()
print('CUDA available: {}'.format(torch.cuda.is_available()), flush=True)
print(f'Crreating retinanet ===>', flush=True)



path_to_weights = '/home/maantonov_1/VKR/weights/efficient/main/2024-03-25/gamma2_alpha0.01/2024-03-25_efficient_small+main_lr:0.0001_step:10_gamma:2_alpha:0.01_n10_m:0.39_f:0.46_val:0.6399.pt'


model = torch.load(path_to_weights, map_location=device)



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






        


model.train()

time_running = []
prediction = []

for i in range(len(test_dataset)):
    print(f'{i} / {len(test_dataset)}', end=' ')
            
    with torch.no_grad():
        
        data = test_dataset[i]
        
        img = data['img'].to(device).float() 
        annot = data['annot']
        
        t = time.time()
        features, regression, classification, anchors = model(img.permute(2, 0, 1).to(device).float().unsqueeze(dim=0))
        t1 = time.time()
 
        t2 = time.time()
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()
        pred_dict = postprocess(img.permute(2, 0, 1).to(device).float().unsqueeze(dim=0),
                        anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        threshold = 0.05, iou_threshold = 0.2)
        t3 = time.time()


        
        print(f'predict: {t1 - t}ms, nms: {t3 - t2}')
        time_running.append(t3-t)
        
    
    pred_dict = dict_list_to_dict(pred_dict)
        
    pred_dict['labels'] = pred_dict['labels'] - 1
        
    
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
    
        
print(f'{datetime.date.today().isoformat()}')
# print(f'path_to_weights {path_to_weights}')
print(f'AVG time: {np.mean(time_running)}')

score_threshold = 0.01
map_score, Fscore = evaluate(prediction, score_threshold = score_threshold)
print(f'score_threshold: {score_threshold}')
print(f'map_score: {map_score}')
print(f'Fscore: {Fscore}')

score_threshold = 0.05
map_score, Fscore = evaluate(prediction, score_threshold = score_threshold)
print(f'score_threshold: {score_threshold}')
print(f'map_score: {map_score}')
print(f'Fscore: {Fscore}')


score_threshold = 0.1
map_score, Fscore = evaluate(prediction, score_threshold = score_threshold)
print(f'score_threshold: {score_threshold}')
print(f'map_score: {map_score}')
print(f'Fscore: {Fscore}')


score_threshold = 0.2
map_score, Fscore = evaluate(prediction, score_threshold = score_threshold)
print(f'score_threshold: {score_threshold}')
print(f'map_score: {map_score}')
print(f'Fscore: {Fscore}')


score_threshold = 0.3
map_score, Fscore = evaluate(prediction, score_threshold = score_threshold)
print(f'score_threshold: {score_threshold}')
print(f'map_score: {map_score}')
print(f'Fscore: {Fscore}')


score_threshold = 0.4
map_score, Fscore = evaluate(prediction, score_threshold = score_threshold)
print(f'score_threshold: {score_threshold}')
print(f'map_score: {map_score}')
print(f'Fscore: {Fscore}')

score_threshold = 0.5
map_score, Fscore = evaluate(prediction, score_threshold = score_threshold)
print(f'score_threshold: {score_threshold}')
print(f'map_score: {map_score}')
print(f'Fscore: {Fscore}')

score_threshold = 0.6
map_score, Fscore = evaluate(prediction, score_threshold = score_threshold)
print(f'score_threshold: {score_threshold}')
print(f'map_score: {map_score}')
print(f'Fscore: {Fscore}')

score_threshold = 0.7
map_score, Fscore = evaluate(prediction, score_threshold = score_threshold)
print(f'score_threshold: {score_threshold}')
print(f'map_score: {map_score}')
print(f'Fscore: {Fscore}')
        
            
    