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
import torchvision
import albumentations as A
from torchvision.ops import nms

from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

from utils.datasetLADD import LADD
from utils.dataloader import collater_annot, ToTorch, Augmenter, Normalizer, Resizer
from utils.Dali_resize import get_dali_pipeline, get_dali_pipeline_aug, flip_bboxes2, flip_bboxes, resize_bb, get_dali_pipeline_two_stages
from utils.metrics import evaluate
# from retinanet import model_oan
from retinanet import losses
from retinanet.center_utils import make_hm_regr, pool, pred2box, pred2centers, get_true_centers, calculate_accuracy_metrics


# import torch.autograd
# torch.autograd.set_detect_anomaly(True)



batch_size = 1
num_workers = 2
resize_to = (1024, 1024)
bb_pad = 0.0
window_size = 640

conf_level1 = 0.0
conf_level2 = 0.95
nms_coef = 0.01
print(f'resize_to: {resize_to[0]}')
print(f'window_size: {window_size}')
print(f'conf_level1: {conf_level1}')
print(f'conf_level2: {conf_level2}')
print(f'nms_coef: {nms_coef}')





path_to_weights1 = '/home/maantonov_1/VKR/weights/retinanet/resize/test/small/583_8/retinanet_resize_583_8_h:1024_w:1024_last.pt'
# path_to_weights1 = '/home/maantonov_1/VKR/weights/retinanet/resize/test/small/583_9/retinanet_resize_583_9_h:1024_w:1024_last.pt'
path_to_weights1 = '/home/maantonov_1/VKR/weights/retinanet/resize/test/small/574/retinanet_resize_574_h:1312_w:1312_last.pt'
# path_to_weights1 = '/home/maantonov_1/VKR/weights/retinanet/resize/test/small/902/retinanet_resize_902_h:1024_w:1024.pt'

path_to_weights2 = '/home/maantonov_1/VKR/weights/faster/main/2024-03-24/2024-03-24_faster_main_lr:0.0003_step:5_n29_m:0.57_f:0.55_val:0.9673.pt'
# path_to_weights2 = '/home/maantonov_1/VKR/weights/faster/main/2024-05-03/2024-05-03_faster_main_lr:0.0001_step:30.pt'
# path_to_weights2 = '/home/maantonov_1/VKR/weights/retinanet/test/main/104/retinanet_oan_104.pt'
# path_to_weights2 = '/home/maantonov_1/VKR/weights/retinanet/test/small/101/retinanet_oan_101_last.pt'




main_dir = '/home/maantonov_1/VKR/data/main_data/test/'
images_dir_val = main_dir + 'images'
annotations_file_val = main_dir + 'annotations/annot.json'
# annotations_file_val = main_dir + 'true_annotations/annot.json'


dali_iterator_two_stages = DALIGenericIterator(
    pipelines=[get_dali_pipeline_two_stages(images_dir = images_dir_val, annotations_file = annotations_file_val,resize_dims = resize_to, batch_size = 1, num_threads = num_workers)],
    output_map=['images', 'images_resized', 'bboxes', 'img_id'],
    reader_name='Reader',
    last_batch_policy=LastBatchPolicy.PARTIAL,
    auto_reset=True,
    dynamic_shape=True
)

print(f'dataset Created', flush=True)


def get_true_centers(pred, scores, dist = 10):
    x = pred[:, 0]
    y = pred[:, 1]

    indices = np.argsort(-scores)
    sorted_points = np.column_stack((x[indices], y[indices]))
    sorted_scores = scores[indices]

    distance_threshold = dist

    keep = np.ones(len(scores), dtype=bool)

    for i in range(len(sorted_points)):
        if not keep[indices[i]]:
            continue
        distances = np.sqrt(np.sum((sorted_points[i] - sorted_points)**2, axis=1))
        for j in range(i + 1, len(sorted_points)):
            if distances[j] < distance_threshold and sorted_scores[j] < sorted_scores[i]:
                keep[indices[j]] = False

    filtered_points = sorted_points[keep[indices]]
    filtered_scores = sorted_scores[keep[indices]]
    return filtered_points, filtered_scores




device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()
print('CUDA available: {}'.format(torch.cuda.is_available()), flush=True)
print(f'Crreating retinanet ===>', flush=True)


retinanet = torch.load(path_to_weights1, map_location=device)
retinanet.to(device)
retinanet.eval()


weights2 = '/home/maantonov_1/VKR/weights/faster/from_slast/small_model.ckpt'
second_model = torch.load(path_to_weights2, map_location=device)

weights = torch.load(weights2, map_location=device)
print(second_model.load_state_dict(weights, strict=True))
# second_model.roi_heads.nms_thresh = 0.1

second_model.to(device)
second_model.eval()


active_list = []
prediction = []
time_running = []
precision_list = []
recall_list = [] 
f1_list = []

for iter_num, data in enumerate(dali_iterator_two_stages):
            
    # print(f'{iter_num} / {len(dali_iterator_two_stages)}', end='')
    with torch.no_grad():
        
        img = data[0]['images']/255
        images_resized = data[0]['images_resized']
        annot = data[0]['bboxes']
        img_id = data[0]['img_id']
        # print(img.shape)
        # print(images_resized.shape)
        _,_,H,W = img.shape
        
        t = time.time()
        pred = retinanet(images_resized)
        t1 = time.time()
        
        time_running.append(t1-t)
        # print(f'\n    time: {t1-t}')
        
        scores, labels, boxes = pred[:3]
        
        
        
        filer = scores > conf_level1
        scores = scores[filer]
        boxes = boxes[filer]
        labels = labels[filer]
           
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy() * 0
        boxes  = boxes.cpu().numpy() 
        
        if len(scores) != 0:

            centers = np.zeros((boxes.shape[0],2))
            centers[:,0] = (boxes[:,2] + boxes[:,0])//2
            centers[:,1] = (boxes[:,3] + boxes[:,1])//2

            centers, scores = get_true_centers(centers, scores, dist = 50)
            centers = torch.tensor(centers)
            
            centers[:,0] = torch.clip(torch.clip(centers[:,0] * W / resize_to[1] - window_size // 2, 0, W) + window_size // 2, 0, W)
            centers[:,1] = torch.clip(torch.clip(centers[:,1] * H / resize_to[0] - window_size // 2, 0, H) + window_size // 2, 0, H)
            centers[:,0] = torch.clip(torch.clip(centers[:,0] + window_size // 2, 0, W) - window_size // 2, 0, W)
            centers[:,1] = torch.clip(torch.clip(centers[:,1] + window_size // 2, 0, H) - window_size // 2, 0, H)
            
            images_to_pred = []

            
            
            
            boxes  = torch.tensor([])
            labels = torch.tensor([])
            scores = torch.tensor([])
            
            for center in centers:
                x = center[0]
                y = center[1]
                x_min = int(x - window_size //2)
                y_min = int(y - window_size //2)
                x_max = int(x + window_size //2)
                y_max = int(y + window_size //2)
                # images_to_pred.append([y_min, y_max, x_min, x_max])
                
                pred = second_model(img[:,:,y_min:y_max, x_min:x_max])
                
                if len(pred[0]['labels']) == 0:
                    continue
                pred[0]['boxes'][:, 0:3:2] += x_min
                pred[0]['boxes'][:, 1:4:2] += y_min
                
                boxes = torch.cat((boxes, pred[0]['boxes'].cpu()), dim=0)
                labels = torch.cat((labels, pred[0]['labels'].cpu()), dim=0)
                scores = torch.cat((scores, pred[0]['scores'].cpu()), dim=0)
                
                #score labels boxes
                # if len(pred[0]) == 0:
                #     continue
                # pred[2][:, 0:3:2] += x_min
                # pred[2][:, 1:4:2] += y_min
                
                # boxes = torch.cat((boxes, pred[2].cpu()), dim=0)
                # labels = torch.cat((labels, pred[1].cpu()), dim=0)
                # scores = torch.cat((scores, pred[0].cpu()), dim=0)
            
            if len(scores) != 0:
                
                filer = scores > conf_level2
                scores = scores[filer]
                boxes = boxes[filer]
                labels = labels[filer]
                
                keep = nms(boxes, scores, nms_coef)
            
                scores = scores[keep].cpu().numpy()
                labels = labels[keep].cpu().numpy() * 0
                boxes  = boxes[keep].cpu().numpy() 
            else:
                scores = scores.cpu().numpy()
                labels = labels.cpu().numpy() * 0
                boxes  = boxes.cpu().numpy() 
        
        
    
    
    pred_dict = {} 
    pred_dict['scores'] = scores
    pred_dict['labels'] = labels
    pred_dict['boxes']  = boxes
    
    
    active_annot = annot.squeeze(0).cpu().numpy()
    
    gt_dict = {}
    if active_annot.shape[0] != 0:
        gt_labels = np.zeros(active_annot.shape[0]) 
        gt_boxes  = active_annot
    else:
        gt_labels = np.array([])
        gt_boxes  = np.array([])
    gt_dict['boxes'] = gt_boxes
    gt_dict['labels'] = gt_labels
    

    
    # if active_annot_tensor.shape[0] != 0:
    #     x = (active_annot_tensor[:,2] + active_annot_tensor[:,0])//2
    #     y = (active_annot_tensor[:,3] + active_annot_tensor[:,1])//2
    #     true_centers = np.column_stack((x, y))
    # else:
    #     true_centers = []
    
    # if boxes.shape[0] != 0:
    #     x = (boxes[:,2] + boxes[:,0])//2
    #     y = (boxes[:,3] + boxes[:,1])//2
    #     pred_center = np.column_stack((x, y))
    # else:
    #     pred_center = []
    
    
    # precision, recall, f1 = calculate_accuracy_metrics(pred_center, true_centers, threshold = 50)
    
    
    # precision_list.append(precision)
    # recall_list.append(recall) 
    # f1_list.append(f1)

    map_score, Fscore = evaluate([(gt_dict, pred_dict)], score_threshold = 0.05)

    if len(scores) == 0:
         active_list.append([float(img_id), -1, -1, -1, -1, -1, map_score, Fscore])
    for j in range(scores.shape[0]):
        active_list.append([float(img_id), float(scores[j]), float(boxes[j][0]), float(boxes[j][1]), float(boxes[j][2]), float(boxes[j][3]), map_score, Fscore])
    
    # print([float(img_id), float(scores[j]), float(boxes[j][0]), float(boxes[j][1]), float(boxes[j][2]), float(boxes[j][3]), map_score, Fscore])

    prediction.append((gt_dict, pred_dict))
    
    
pd.DataFrame(active_list, columns = ['id','sccore','xmin','ymin','xmax','ymax','map','fscore']).to_csv('/home/maantonov_1/VKR/actual_scripts/retinanet_sep/AL/AL_on_test_two_stages.csv')
    

print(f'{datetime.date.today().isoformat()}')
print(f'path_to_weights1 {path_to_weights1}')
print(f'path_to_weights2 {path_to_weights2}')
print(f'AVG time: {np.mean(time_running)}')

iou_thr_max=0.6
print(f'iou_thr_max: {iou_thr_max}')

score_threshold = 0.05
map_score, Fscore = evaluate(prediction, score_threshold = score_threshold, iou_thr_max=iou_thr_max)
print(f'score_threshold: {score_threshold}')
print(f'map_score: {map_score}')
print(f'Fscore: {Fscore}')

print('__________________')

# print(f'precision {np.mean(precision_list)}')
# print(f'recall {np.mean(recall_list)}')
# print(f'f1 {np.mean(f1_list)}')

# print('__________________')

iou_thr_max=0.95
print(f'iou_thr_max: {iou_thr_max}')

score_threshold = 0.01
map_score, Fscore = evaluate(prediction, score_threshold = score_threshold, iou_thr_max=iou_thr_max)
print(f'score_threshold: {score_threshold}')
print(f'map_score: {map_score}')
print(f'Fscore: {Fscore}')

score_threshold = 0.05
map_score, Fscore = evaluate(prediction, score_threshold = score_threshold, iou_thr_max=iou_thr_max)
print(f'score_threshold: {score_threshold}')
print(f'map_score: {map_score}')
print(f'Fscore: {Fscore}')


score_threshold = 0.1
map_score, Fscore = evaluate(prediction, score_threshold = score_threshold, iou_thr_max=iou_thr_max)
print(f'score_threshold: {score_threshold}')
print(f'map_score: {map_score}')
print(f'Fscore: {Fscore}')


score_threshold = 0.2
map_score, Fscore = evaluate(prediction, score_threshold = score_threshold, iou_thr_max=iou_thr_max)
print(f'score_threshold: {score_threshold}')
print(f'map_score: {map_score}')
print(f'Fscore: {Fscore}')


score_threshold = 0.3
map_score, Fscore = evaluate(prediction, score_threshold = score_threshold, iou_thr_max=iou_thr_max)
print(f'score_threshold: {score_threshold}')
print(f'map_score: {map_score}')
print(f'Fscore: {Fscore}')


score_threshold = 0.4
map_score, Fscore = evaluate(prediction, score_threshold = score_threshold, iou_thr_max=iou_thr_max)
print(f'score_threshold: {score_threshold}')
print(f'map_score: {map_score}')
print(f'Fscore: {Fscore}')


score_threshold = 0.5
map_score, Fscore = evaluate(prediction, score_threshold = score_threshold, iou_thr_max=iou_thr_max)
print(f'score_threshold: {score_threshold}')
print(f'map_score: {map_score}')
print(f'Fscore: {Fscore}')

score_threshold = 0.6
map_score, Fscore = evaluate(prediction, score_threshold = score_threshold, iou_thr_max=iou_thr_max)
print(f'score_threshold: {score_threshold}')
print(f'map_score: {map_score}')
print(f'Fscore: {Fscore}')

score_threshold = 0.7
map_score, Fscore = evaluate(prediction, score_threshold = score_threshold, iou_thr_max=iou_thr_max)
print(f'score_threshold: {score_threshold}')
print(f'map_score: {map_score}')
print(f'Fscore: {Fscore}')
            
    
    
