import os
import time
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
import torchvision.transforms as T
from torchvision.utils import make_grid 
from torch.utils.data import DataLoader

from utils.datasetLADD import LADD
from utils.dataloader import ToTorch, Augmenter, Normalizer, Resizer
from retinanet import model_oan
from metrics import evaluate



test_df = [{'dataframe': pd.read_csv('/home/maantonov_1/VKR/data/main_data/test/test_main.csv'),
             'image_dir': '/home/maantonov_1/VKR/data/main_data/test/images'}]

score_threshold = 0.5

# test_df = [{'dataframe': pd.read_csv('/home/maantonov_1/VKR/data/crope_data/main/crop_val_1024/crop_val_1024.csv'),
#              'image_dir': '/home/maantonov_1/VKR/data/crope_data/main/crop_val_1024/images'}]


test_dataset = LADD(test_df, mode = "valid", smart_crop = True, new_shape = (1024,1024), transforms = T.Compose([Normalizer(), Resizer()]))

print(f'dataset Created', flush=True)


def transform_annotations(annots):

    if annots.shape[0] != 0:
        img_annots = {
            'boxes': annots[:, :4].to(device).long(),  
            'labels': torch.tensor([0] * annots.shape[0]).to(device).long(),  
        }
    else:
        img_annots = {
            'boxes': torch.tensor([]).to(device).long(),  
            'labels': torch.tensor([]).to(device).long(),  
        }

    
    return img_annots

def to_numpy(dic):
    for key in dic:
        dic[key] = dic[key].cpu().numpy()


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()
print('CUDA available: {}'.format(torch.cuda.is_available()), flush=True)
print(f'Crreating retinanet ===>', flush=True)


path_to_weights = '/home/maantonov_1/VKR/weights/faster_main/2024-03-15/2024-03-15_faster_main_lr:0.0003_step:5_n13_m:0.53_f:0.34_val:0.0294.pt'

model = torch.load(path_to_weights, map_location=device)

if torch.cuda.is_available():
    model = torch.nn.DataParallel(model).cuda()


model.training = False
model.eval()

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
    print(f'{i} / {len(test_dataset)}', end=' ')
    
    
            
    with torch.no_grad():
        
        data = test_dataset[i]
        
        img = data['img'].to(device).float() 
        annot = data['annot']
        
        t = time.time()
        annot = transform_annotations(annot)
        
        pred_dict = model(img.permute(2,0,1).unsqueeze(0).to(device))
        t1 = time.time()
        time_running.append(t1 - t)
        print(f'time: {t1 - t}')
        
        pred_dict[0]['labels'] = pred_dict[0]['labels'] * 0
        
        to_numpy(annot)
        to_numpy(pred_dict[0])
    

    prediction.append((annot, pred_dict[0]))
    
    
map_score, Fscore = evaluate(prediction, score_threshold = score_threshold)

print(f'path_to_weights {path_to_weights}')
print(f'score_threshold: {score_threshold}')
print(f'map_score: {map_score}')
print(f'Fscore: {Fscore}')
print(f'AVG time: {np.mean(time_running)}')