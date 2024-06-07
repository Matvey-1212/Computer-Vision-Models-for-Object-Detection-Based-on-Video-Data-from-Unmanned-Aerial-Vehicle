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
from utils.Dali_resize import get_dali_pipeline, get_dali_pipeline_aug, flip_bboxes2, flip_bboxes, resize_bb
from utils.metrics import evaluate



# import torch.autograd
# torch.autograd.set_detect_anomaly(True)




batch_size = 1
num_workers = 2
resize_to = (1024, 1024)



# path_to_weights = '/home/maantonov_1/VKR/weights/faster/small/2024-03-24/2024-03-24_faster_main_lr:0.0003_step:5_n28_m:0.16_f:0.25_val:2.1593.pt'
# path_to_weights = '/home/maantonov_1/VKR/weights/faster/resize/main/2024-04-12/2024-04-12_faster_resize_h:1024_w:1024_small+main_lr:0.0003_step:10_n29_m:0.51_f:0.31_val:0.0360.pt'
path_to_weights = '/home/maantonov_1/VKR/weights/faster/resize/main/2024-04-11/2024-04-11_faster_resize_h:1024_w:1024_small+main_lr:0.0003_step:10_n29_m:0.61_f:0.43_val:0.0293.pt'


main_dir = '/home/maantonov_1/VKR/data/main_data/test/'
images_dir_val = main_dir + 'images'
annotations_file_val = main_dir + 'annotations/annot.json'




dali_iterator_val = DALIGenericIterator(
    pipelines=[get_dali_pipeline(images_dir = images_dir_val, annotations_file = annotations_file_val, resize_dims = resize_to,batch_size = batch_size, num_threads = num_workers)],
    output_map=['data', 'res_images', 'bboxe', 'bbox_shapes', 'img_id','original_sizes'],
    reader_name='Reader',
    last_batch_policy=LastBatchPolicy.PARTIAL,
    auto_reset=True,
    dynamic_shape=True
)

print(f'dataset Created', flush=True)



def transform_annotations(batch_imgs, batch_annots):
    transformed_annotations = []

    for i in range(batch_annots.shape[0]): 
        annots = batch_annots[i].to(device)
        valid_annots = annots[annots[:, 3] != -1]  

        img_annots = {
            'boxes': valid_annots[:, :4].to(device).long(),  
            'labels': torch.tensor([1] * valid_annots.shape[0]).to(device).long(),  
        }
        transformed_annotations.append(img_annots)
    return batch_imgs, transformed_annotations

def to_numpy(dic):
    for key in dic:
        dic[key] = dic[key].cpu().numpy()
        
        



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()
print('CUDA available: {}'.format(torch.cuda.is_available()), flush=True)
print(f'Crreating model ===>', flush=True)





model = torch.load(path_to_weights, map_location=device)
path_to_weights = '/home/maantonov_1/VKR/weights/faster/from_slast/big_model.ckpt'
weights = torch.load(path_to_weights,map_location=torch.device('cpu'))
print(model.load_state_dict(weights, strict=True))

model.roi_heads.nms_thresh = 0.1

model.to(device)

def dict_list_to_dict(list_of_dicts):
    concatenated_dict = {}
    for key in list_of_dicts[0]:
        concatenated_dict[key] = torch.cat([d[key] for d in list_of_dicts], dim=0)
    return concatenated_dict


    

model.eval()

time_running = []
prediction = []

active_list = []

for iter_num, data in enumerate(dali_iterator_val):
    print(f'{iter_num} / {len(dali_iterator_val)}')
    with torch.no_grad():
        
        img = data[0]['data']/255
        bbox = data[0]['bboxe'].int()
        img_id = data[0]['img_id']
        bb_shape = data[0]['bbox_shapes'].cpu()
        original_sizes = data[0]['original_sizes']
        # bbox = resize_bb(bbox, original_sizes, bb_pad = 0.1, new_shape = resize_to)

        
        img, annot = transform_annotations(img, bbox)
        try:
            t = time.time()
            pred_dict = model(img)
            t1 = time.time()
            time_running.append(t1-t)
        except:
            print(z)
            continue
    
    scores = pred_dict[0]['scores']
    boxes  = pred_dict[0]['boxes']
    
    annot, pred_dict = dict_list_to_dict(annot), dict_list_to_dict(pred_dict)
        
    pred_dict['labels'] = pred_dict['labels'] * 0
    annot['labels'] = annot['labels'] * 0
        
    to_numpy(annot)
    to_numpy(pred_dict)

    map_score, Fscore = evaluate([(annot, pred_dict)], score_threshold = 0.05)

    if len(scores) == 0:
        active_list.append([float(img_id), -1, -1, -1, -1, -1, map_score, Fscore])
    for j in range(scores.shape[0]):
        active_list.append([float(img_id), float(scores[j]), float(boxes[j][0]), float(boxes[j][1]), float(boxes[j][2]), float(boxes[j][3]), map_score, Fscore])
    
    # print([float(img_id), float(scores[j]), float(boxes[j][0]), float(boxes[j][1]), float(boxes[j][2]), float(boxes[j][3]), map_score, Fscore])

    prediction.append((annot, pred_dict))
    
    
pd.DataFrame(active_list, columns = ['id','sccore','xmin','ymin','xmax','ymax','map','fscore']).to_csv('/home/maantonov_1/VKR/actual_scripts/faster_pytorch/AL/AL_on_test.csv')
    
print(f'{datetime.date.today().isoformat()}')
# print(f'path_to_weights {path_to_weights}')
print(f'AVG time: {np.mean(time_running)}')

iou_thr_max=0.6
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
            
        
            
     
    
            
  