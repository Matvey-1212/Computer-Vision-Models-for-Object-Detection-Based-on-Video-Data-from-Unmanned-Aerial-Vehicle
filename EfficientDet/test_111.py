import os
import time
import numpy as np
import pandas as pd
import datetime

import torch
import torch.optim as optim
import torchvision.transforms as T
from torchvision.utils import make_grid 
from torch.utils.data import DataLoader

from utils.datasetLADD import LADD
from utils.dataloader import ToTorch, Augmenter, Normalizer, Resizer
from retinanet import model_oan
from metrics import evaluate

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



# path_to_weights = '/home/maantonov_1/VKR/weights/retinanet/10_03_2024/retinanet_oan_vis+small_lr+ful_lr0.0003_step5_gamma2_alpha0.1_12_0.48438382037660715_0.5626141205112599.pt'
path_to_weights = '/home/maantonov_1/VKR/weights/retinanet/main/2024-03-24/gamma2_alpha0.01/2024-03-24_retinanet_oan_vis+small_lr:0.0003_step:10_gamma:2_alpha:0.01_n24_m:0.56_f:0.48_val:0.1731.pt'

retinanet = torch.load(path_to_weights, map_location=device)

# retinanet = model_oan.resnet50(num_classes = 2, pretrained = False, inputs = 3)

if torch.cuda.is_available():
    retinanet = torch.nn.DataParallel(retinanet).cuda()


retinanet.training = False
retinanet.eval()

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
    
prediction = []
time_running = []

for i in range(len(test_dataset)):
    print(f'{i} / {len(test_dataset)}', end='')
    
    
            
    with torch.no_grad():
        
        data = test_dataset[i]
        
        img = data['img'].to(device).float() 
        annot = data['annot']
        
        t = time.time()
        scores,  labels, boxes = retinanet(img.permute(2, 0, 1).to(device).float().unsqueeze(dim=0)) #labels,
        t1 = time.time()
        
    time_running.append(t1-t)
    print(f'\n    time: {t1-t}')
    
    pred_dict = {}
    scores = scores.cpu().numpy()
    labels = labels.cpu().numpy()
    boxes  = boxes.cpu().numpy()
    pred_dict['scores'] = scores
    pred_dict['labels'] = labels
    pred_dict['boxes']  = boxes
    
    
    gt_dict = {}
    if annot.shape[0] != 0:
        gt_labels = annot[:,-1].numpy()
        gt_boxes  = annot[:,:-1].numpy()
    else:
        gt_labels = np.array([])
        gt_boxes  = np.array([])
    gt_dict['boxes'] = gt_boxes
    gt_dict['labels'] = gt_labels
    

    prediction.append((gt_dict, pred_dict))
    if i >= 3:
        print(gt_dict)
        print(pred_dict)
        break
    
map_score, Fscore = evaluate(prediction, score_threshold = score_threshold)

print(f'{datetime.date.today().isoformat()}')
print(f'path_to_weights {path_to_weights}')
print(f'score_threshold: {score_threshold}')
print(f'map_score: {map_score}')
print(f'Fscore: {Fscore}')
print(f'AVG time: {np.mean(time_running)}')