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

from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

from utils.datasetLADD import LADD
from utils.dataloader import collater_annot, ToTorch, Augmenter, Normalizer, Resizer
from utils.Dali_resize import get_dali_pipeline, get_dali_pipeline_aug, flip_bboxes2, flip_bboxes, resize_bb
from utils.metrics import evaluate
from retinanet import model_oan, model_incpetion
from retinanet import losses


# import torch.autograd
# torch.autograd.set_detect_anomaly(True)



batch_size = 1
num_workers = 2
bb_pad = 0.5
resize_to = (1312, 1312)








# path_to_weights = '/home/maantonov_1/VKR/weights/retinanet/resize/main/2024-03-24/gamma2_alpha0.01/2024-03-24_retinanet_oan_resize_h:1024_w:1024_vis+small+main_lr:0.0003_step:10_gamma:2_alpha:0.01_n29_m:0.31_f:0.46_val:0.1128_last.pt'
# path_to_weights = '/home/maantonov_1/VKR/weights/retinanet/resize/main/2024-03-24/gamma2_alpha0.01/2024-03-24_retinanet_oan_resize_h:1024_w:1024_vis+small+main_lr:0.0003_step:10_gamma:2_alpha:0.01_n21_m:0.29_f:0.49_val:0.1131.pt'
# path_to_weights = '/home/maantonov_1/VKR/weights/retinanet/resize/main_more_sum/2024-04-11/gamma2_alpha0.01/2024-04-11_retinanet_oan_resize_h:1024_w:1024_vis+small+main_lr:0.0003_step:10_gamma:2_alpha:0.01_n26_m:0.30_f:0.56_val:0.0762.pt'
# path_to_weights = '/home/maantonov_1/VKR/weights/retinanet/resize/test/main/1713651785/retinanet_resize_h:1312_w:1312_main.pt'
# path_to_weights = '/home/maantonov_1/VKR/weights/retinanet/resize/test/main/1713683844/retinanet_resize_1713683844_h:1312_w:1312_main.pt'


path_to_weights = '/home/maantonov_1/VKR/weights/retinanet/resize/test/main/1713810939/retinanet_resize_1713810939_h:1312_w:1312_main.pt'


# main_dir = '/home/maantonov_1/VKR/data/crope_data/main/crop_val_1024/'
# main_dir = '/home/maantonov_1/VKR/data/crope_data/small/small_crop_val/'
# main_dir = '/home/maantonov_1/VKR/data/main_data/train/'
# main_dir = '/home/maantonov_1/VKR/data/small_train/train/'
main_dir = '/home/maantonov_1/VKR/data/main_data/test/'
images_dir_val = main_dir + 'images'
annotations_file_val = main_dir + 'annotations/annot.json'

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
print(f'Crreating retinanet ===>', flush=True)

# retinanet = model_oan.resnet50(num_classes = 2, pretrained = False, inputs = 3)


retinanet = torch.load(path_to_weights, map_location=device)

retinanet.to(device)


retinanet.train()

epoch_loss = []
epoch_loss_cr = []
prediction = []
time_running = []

for iter_num, data in enumerate(dali_iterator_val):
            
    # print(f'{iter_num} / {len(dali_iterator_val)}', end='')
    with torch.no_grad():
            
        img = data[0]['data']
        bbox = data[0]['bboxe'].int()
        z = data[0]['img_id']
        bb_shape = data[0]['bbox_shapes']
        original_sizes = data[0]['original_sizes']
        bbox = resize_bb(bbox, original_sizes, bb_pad = bb_pad, new_shape = resize_to)
    
        new_elements = torch.where(bbox[:, :, 3] == -1, -1, 1).unsqueeze(2)  
        annot = torch.cat((bbox, new_elements), dim=2).to(device)
        
        classification_loss, regression_loss = retinanet([img, annot])


        loss = classification_loss + regression_loss #+ 4 * oan_loss #+ class_loss_ap #+ reg_loss_ap
        oan_loss = 0

        epoch_loss_cr.append([float(classification_loss), float(regression_loss)])
        epoch_loss.append(float(loss))

        print(
            ' Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                iter_num, float(classification_loss), float(regression_loss), np.mean(epoch_loss)), flush=True)
    

pd.DataFrame(epoch_loss_cr, columns = ['c','r']).to_csv('/home/maantonov_1/VKR/actual_scripts/retinanet_sep/AL/AL_on_test.csv')
