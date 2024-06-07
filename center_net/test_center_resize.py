import os
import time
import numpy as np
import pandas as pd
import random
import datetime
import math

import sys
import yaml

import wandb
import torch
import torch.nn as nn
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
from utils.Dali_resize import get_dali_pipeline, get_dali_pipeline_aug, flip_bboxes2, flip_bboxes, resize_bb, rotate_bboxes
from utils.metrics import evaluate
from centers import model_center
from centers import losses
from centers import aploss
from centers.center_utils import make_hm_regr, pool, pred2box, pred2centers, get_true_centers, calculate_accuracy_metrics



# import torch.autograd
# torch.autograd.set_detect_anomaly(True)



batch_size = 1
num_workers = 2
resize_to = (1024, 1024)



input_size = 1024
IN_SCALE = 1024//resize_to[0] 
MODEL_SCALE = 4




# path_to_weights = '/home/maantonov_1/VKR/weights/retinanet/resize/main/2024-03-24/gamma2_alpha0.01/2024-03-24_retinanet_oan_resize_h:1024_w:1024_vis+small+main_lr:0.0003_step:10_gamma:2_alpha:0.01_n29_m:0.31_f:0.46_val:0.1128_last.pt'
# path_to_weights = '/home/maantonov_1/VKR/weights/retinanet/resize/main/2024-03-24/gamma2_alpha0.01/2024-03-24_retinanet_oan_resize_h:1024_w:1024_vis+small+main_lr:0.0003_step:10_gamma:2_alpha:0.01_n21_m:0.29_f:0.49_val:0.1131.pt'
# path_to_weights = '/home/maantonov_1/VKR/weights/retinanet/resize/main_more_sum/2024-04-11/gamma2_alpha0.01/2024-04-11_retinanet_oan_resize_h:1024_w:1024_vis+small+main_lr:0.0003_step:10_gamma:2_alpha:0.01_n26_m:0.30_f:0.56_val:0.0762.pt'
# path_to_weights = '/home/maantonov_1/VKR/weights/retinanet/resize/test/main/1713651785/retinanet_resize_h:1312_w:1312_main.pt'
# path_to_weights = '/home/maantonov_1/VKR/weights/retinanet/resize/test/main/1713683844/retinanet_resize_1713683844_h:1312_w:1312_main.pt'


path_to_weights = '/home/maantonov_1/VKR/weights/center/resize/test/small/201/center_resize_201_h:1024_w:1024.pt'


# main_dir = '/home/maantonov_1/VKR/data/crope_data/main/crop_val_1024/'
# main_dir = '/home/maantonov_1/VKR/data/crope_data/small/small_crop_val/'
# main_dir = '/home/maantonov_1/VKR/data/main_data/train/'
# main_dir = '/home/maantonov_1/VKR/data/small_train/train/'
main_dir = '/home/maantonov_1/VKR/data/main_data/test/'
images_dir_val = main_dir + 'images'
# annotations_file_val = main_dir + 'annotations/annot.json'
annotations_file_val = main_dir + 'true_annotations/annot.json'

# main_dir = '/home/maantonov_1/VKR/data/main_data/train/'
# # main_dir = '/home/maantonov_1/VKR/data/small_train/train/'
# images_dir_val = main_dir + 'images'
# annotations_file_val = main_dir + 'val_annot/annot.json'




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

# retinanet = model_oan.resnet50(num_classes = 2, pretrained = False, inputs = 3)


model = torch.load(path_to_weights, map_location=device)

model.to(device)


model.eval()

active_list = []

prediction = []
time_running = []

precision_list = []
recall_list = [] 
f1_list = []



for iter_num, data in enumerate(dali_iterator_val):
            
    print(f'{iter_num} / {len(dali_iterator_val)}', end='')
    with torch.no_grad():
        
        img = data[0]['data']
        bbox = data[0]['bboxe']
        img_id = data[0]['img_id']
        bb_shape = data[0]['bbox_shapes'].cpu()
        original_sizes = data[0]['original_sizes']
        bbox = resize_bb(bbox, original_sizes, bb_pad = 0.5, new_shape = resize_to).int()
        
        hm_gt, regr_gt = make_hm_regr(bbox, bb_shape)
        
        new_elements = torch.where(bbox[:, :, 3] == -1, -1, 1).unsqueeze(2)  
        annot = torch.cat((bbox, new_elements), dim=2).cpu()
        
        t = time.time()
        hm, regr = model(img)
        t1 = time.time()
        
        
    # pred = torch.cat((hm, regr), 1)
    # hm, regr = pred
    # for hm, regr in zip(hms, regrs):
    hm = hm.squeeze(0).squeeze(0).cpu().numpy()
    regr = regr.squeeze(0).cpu().numpy()

    hm = torch.sigmoid(torch.from_numpy(hm)).numpy()
    hm = pool(hm)

    scores, pred_center = pred2centers(hm, regr, thresh=0.2)
    
    pred_center = (pred_center[0] * MODEL_SCALE, pred_center[1] * MODEL_SCALE)

    pred_center, scores = get_true_centers(pred_center, scores, dist=15)
    
    active_annot = [annot[i, :bb_shape[i, 0]] for i in range(bb_shape.size(0))]
    active_annot_tensor = torch.cat(active_annot, dim=0)
    
    x = (active_annot_tensor[:,2] + active_annot_tensor[:,0])//2
    y = (active_annot_tensor[:,3] + active_annot_tensor[:,1])//2
    true_centers = np.column_stack((x, y))
    
    precision, recall, f1 = calculate_accuracy_metrics(pred_center, true_centers, threshold = 50)
        
    time_running.append(t1-t)
    print(f'\n    time: {t1-t}')
    
    if len(scores) == 0:
         active_list.append([float(img_id), -1, -1, -1, precision, recall, f1])
    for j in range(scores.shape[0]):
        active_list.append([float(img_id), float(scores[j]), float(pred_center[j][1]), float(pred_center[j][0]), precision, recall, f1])
    
    
    precision_list.append(precision)
    recall_list.append(recall) 
    f1_list.append(f1)
    
pd.DataFrame(active_list, columns = ['id','sccore','xmin','ymin','precision','recall','f1']).to_csv('/home/maantonov_1/VKR/actual_scripts/center_net/AL/AL_on_test.csv')
    
print(f'{datetime.date.today().isoformat()}')
print(f'path_to_weights {path_to_weights}')
print(f'AVG time: {np.mean(time_running)}')


print(f'precision {np.mean(precision_list)}')
print(f'recall {np.mean(recall_list)}')
print(f'f1 {np.mean(f1_list)}')
        
    # pred_dict = {}    
    # time_running.append(t1-t)
    # scores = scores
    # labels = np.zeros(boxes.shape[0])
    # boxes  = boxes
    # pred_dict['scores'] = scores
    # pred_dict['labels'] = labels
    # pred_dict['boxes']  = boxes
    
    
    
    
    # gt_dict = {}
    # if annot.shape[0] != 0:
    #     gt_labels = active_annot_tensor[:,-1].numpy() * 0 
    #     gt_boxes  = active_annot_tensor[:,:-1].numpy()
    # else:
    #     gt_labels = np.array([])
    #     gt_boxes  = np.array([])
    # gt_dict['boxes'] = gt_boxes
    # gt_dict['labels'] = gt_labels

    # map_score, Fscore = evaluate([(gt_dict, pred_dict)], score_threshold = 0.05)

    # if len(scores) == 0:
    #      active_list.append([float(img_id), -1, -1, -1, -1, -1, map_score, Fscore])
    # for j in range(scores.shape[0]):
    #     active_list.append([float(img_id), float(scores[j]), float(boxes[j][0]), float(boxes[j][1]), float(boxes[j][2]), float(boxes[j][3]), map_score, Fscore])
    
    # print([float(img_id), float(scores[j]), float(boxes[j][0]), float(boxes[j][1]), float(boxes[j][2]), float(boxes[j][3]), map_score, Fscore])

    # prediction.append((gt_dict, pred_dict))
    
# pd.DataFrame(active_list, columns = ['id','sccore','xmin','ymin','xmax','ymax','map','fscore']).to_csv('/home/maantonov_1/VKR/actual_scripts/retinanet_sep/AL/AL_on_test_1024.csv')
    

# print(f'{datetime.date.today().isoformat()}')
# print(f'path_to_weights {path_to_weights}')
# print(f'AVG time: {np.mean(time_running)}')

# iou_thr_max=0.6
# print(f'iou_thr_max: {iou_thr_max}')

# score_threshold = 0.05
# map_score, Fscore = evaluate(prediction, score_threshold = score_threshold, iou_thr_max=iou_thr_max)
# print(f'score_threshold: {score_threshold}')
# print(f'map_score: {map_score}')
# print(f'Fscore: {Fscore}')

# print('__________________')

# iou_thr_max=0.95
# print(f'iou_thr_max: {iou_thr_max}')

# score_threshold = 0.01
# map_score, Fscore = evaluate(prediction, score_threshold = score_threshold, iou_thr_max=iou_thr_max)
# print(f'score_threshold: {score_threshold}')
# print(f'map_score: {map_score}')
# print(f'Fscore: {Fscore}')

# score_threshold = 0.05
# map_score, Fscore = evaluate(prediction, score_threshold = score_threshold, iou_thr_max=iou_thr_max)
# print(f'score_threshold: {score_threshold}')
# print(f'map_score: {map_score}')
# print(f'Fscore: {Fscore}')


# score_threshold = 0.1
# map_score, Fscore = evaluate(prediction, score_threshold = score_threshold, iou_thr_max=iou_thr_max)
# print(f'score_threshold: {score_threshold}')
# print(f'map_score: {map_score}')
# print(f'Fscore: {Fscore}')


# score_threshold = 0.2
# map_score, Fscore = evaluate(prediction, score_threshold = score_threshold, iou_thr_max=iou_thr_max)
# print(f'score_threshold: {score_threshold}')
# print(f'map_score: {map_score}')
# print(f'Fscore: {Fscore}')


# score_threshold = 0.3
# map_score, Fscore = evaluate(prediction, score_threshold = score_threshold, iou_thr_max=iou_thr_max)
# print(f'score_threshold: {score_threshold}')
# print(f'map_score: {map_score}')
# print(f'Fscore: {Fscore}')


# score_threshold = 0.4
# map_score, Fscore = evaluate(prediction, score_threshold = score_threshold, iou_thr_max=iou_thr_max)
# print(f'score_threshold: {score_threshold}')
# print(f'map_score: {map_score}')
# print(f'Fscore: {Fscore}')


# score_threshold = 0.5
# map_score, Fscore = evaluate(prediction, score_threshold = score_threshold, iou_thr_max=iou_thr_max)
# print(f'score_threshold: {score_threshold}')
# print(f'map_score: {map_score}')
# print(f'Fscore: {Fscore}')

# score_threshold = 0.6
# map_score, Fscore = evaluate(prediction, score_threshold = score_threshold, iou_thr_max=iou_thr_max)
# print(f'score_threshold: {score_threshold}')
# print(f'map_score: {map_score}')
# print(f'Fscore: {Fscore}')

# score_threshold = 0.7
# map_score, Fscore = evaluate(prediction, score_threshold = score_threshold, iou_thr_max=iou_thr_max)
# print(f'score_threshold: {score_threshold}')
# print(f'map_score: {map_score}')
# print(f'Fscore: {Fscore}')
            
    
    
