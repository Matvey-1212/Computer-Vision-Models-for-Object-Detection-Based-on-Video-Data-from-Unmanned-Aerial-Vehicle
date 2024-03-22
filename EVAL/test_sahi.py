import os
import time
import numpy as np
import pandas as pd
import cv2
import datetime

import torch
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from utils.datasetLADD import LADD
from metrics import evaluate


score_threshold=  0.05
confidence_threshold = 0.3


test_df = [{'dataframe': pd.read_csv('/home/maantonov_1/VKR/data/main_data/test/test_main.csv'),
             'image_dir': '/home/maantonov_1/VKR/data/main_data/test/images'}]


# test_dataset = LADD(test_df, mode = "valid", from_255_to_1 = False, smart_crop = True, new_shape = (1024,1024))
test_dataset = LADD(test_df, mode = "valid", for_sahi = True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()
print('CUDA available: {}'.format(torch.cuda.is_available()), flush=True)

# model = YOLO('/home/maantonov_1/VKR/actual_scripts/yolo/runs/detect/yolov8m_small_data+main2/weights/best.pt')

weights = '/home/maantonov_1/VKR/actual_scripts/yolo/runs/detect/yolov8m_small_data+main2/weights/best.pt'

model = AutoDetectionModel.from_pretrained(
    model_type="yolov8", model_path=weights, confidence_threshold=confidence_threshold, device=device
)

time_running = []
prediction = []


# [({'boxes': np.array([[1321.8750,  274.6667, 1348.8750,  312.6667]]),
#                 'labels': np.array([1])},
#               {'boxes': np.array([[1323.5446,  275.2711, 1350.2203,  315.9069],
#                         [ 119.2671, 1227.5459,  171.1528, 1277.9830],
#                         [ 240.5078, 1147.3656,  270.7879, 1205.0126],
#                         [ 140.9097, 1231.9814,  173.9967, 1285.4724]]),
#                 'scores': np.array([0.9568, 0.3488, 0.1418, 0.0771]),
#                 'labels': np.array([1, 1, 1, 1])}),
#              ({'boxes': np.array([[ 798.7500, 1357.3334,  837.7500, 1396.6666],
#                         [ 829.1250,  777.3333,  873.3750,  818.0000],
#                         [ 886.5000,   34.6667,  916.5000,   77.3333]]),
#                 'labels': np.array([1, 1, 1])},
#               {'boxes': np.array([[ 796.5808, 1354.9255,  836.5349, 1395.8972],
#                         [ 828.8597,  777.9426,  872.5923,  819.8660],
#                         [ 887.7839,   37.1435,  914.8092,   76.3933]]),
#                 'scores': np.array([0.9452, 0.8701, 0.8424]),
#                 'labels': np.array([1, 1, 1])})]


for i in range(len(test_dataset)):
    print(f'{i} / {len(test_dataset)}')
    
    path, annot = test_dataset[i]
    
    t = time.time()
    results = get_sliced_prediction(
        path, model, slice_height=1024, slice_width=1024, overlap_height_ratio=0.2, overlap_width_ratio=0.2
    )
    t1 = time.time()
    
    time_running.append(t1-t)
    print(f'    time: {t1-t}')

    
    object_prediction_list = results.object_prediction_list
    boxes_list = []
    clss_list = []
    scr_list = []
    pred_dict = {}

    for ind, _ in enumerate(object_prediction_list):
        boxes = (
            object_prediction_list[ind].bbox.minx,
            object_prediction_list[ind].bbox.miny,
            object_prediction_list[ind].bbox.maxx,
            object_prediction_list[ind].bbox.maxy,
        )
        clss = object_prediction_list[ind].category.id
        scr = object_prediction_list[ind].score.value
        boxes_list.append(list(boxes))
        clss_list.append(clss)
        scr_list.append(scr)
    pred_dict['boxes'] = np.array(boxes_list)
    pred_dict['labels'] = np.array(clss_list)
    pred_dict['scores'] = np.array(scr_list)
    
    
    gt_dict = {}
    if annot.shape[0] != 0:
        gt_labels = annot[:,-1]
        gt_boxes  = annot[:,:-1]
    else:
        gt_labels = np.array([])
        gt_boxes  = np.array([])
    gt_dict['boxes'] = gt_boxes
    gt_dict['labels'] = gt_labels
    
    
    prediction.append((gt_dict, pred_dict))
    # print((gt_dict, pred_dict))
        

map_score, Fscore = evaluate(prediction, score_threshold = score_threshold)

print(f'{datetime.date.today().isoformat()}')
print(f'score_threshold {score_threshold}')
print(f'confidence_threshold {confidence_threshold}')
print(f'map_score: {map_score}')
print(f'Fscore: {Fscore}')
print(f'AVG time: {np.mean(time_running)}')
        

    
