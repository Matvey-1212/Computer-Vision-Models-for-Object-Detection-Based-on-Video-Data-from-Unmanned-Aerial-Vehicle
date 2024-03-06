import os
import time
import numpy as np
import pandas as pd
import cv2

import torch
from ultralytics import YOLO

from utils.datasetLADD import LADD
from metrics import evaluate





test_df = [{'dataframe': pd.read_csv('/home/maantonov_1/VKR/data/main_data/test/test_main.csv'),
             'image_dir': '/home/maantonov_1/VKR/data/main_data/test/images'}]


test_dataset = LADD(test_df, mode = "valid", season = 0, from_255_to_1 = False, smart_crop = True, new_shape = (640,640))

torch.cuda.empty_cache()

model = YOLO('/home/maantonov_1/VKR/actual_scripts/yolo/runs/detect/yolov8l2/weights/best.pt')

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
    
    data = test_dataset[i]
        
    img = data['img']
    annot = data['annot']
    
    t = time.time()
    res = model.predict(img)
    t1 = time.time()
    
    time_running.append(t1-t)
    print(f'    time: {t1-t}')

    
    boxes_ = res[0].boxes.xyxy.cpu()                        #xywh bbox list
    clss_ = res[0].boxes.cls.cpu().tolist()                 #classes Id list
    confs_ = res[0].boxes.conf.float().cpu().tolist() 
    
    boxes = []
    scores = []
    labels = [] 
    
    for box, cls, conf in zip(boxes_, clss_, confs_):
        boxes.append(box.numpy())
        scores.append(conf)
        labels.append(cls)
    
    
    
    
    pred_dict = {}
    pred_dict['scores'] = np.asarray(scores)
    pred_dict['labels'] = np.asarray(labels)
    pred_dict['boxes']  = np.asarray(boxes)
    
    
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
        

map_score, Fscore = evaluate(prediction, score_threshold = 0.5)

print(f'map_score: {map_score}')
print(f'Fscore: {Fscore}')

        

    
