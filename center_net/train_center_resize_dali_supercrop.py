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
import torch.utils.model_zoo as model_zoo

from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

from utils.datasetLADD import LADD
from utils.dataloader import collater_annot, ToTorch, Augmenter, Normalizer, Resizer
from utils.Dali_resize import get_dali_pipeline, get_dali_pipeline_aug, flip_bboxes2, flip_bboxes, resize_bb, rotate_bboxes
from utils.metrics import evaluate
from centers import model_center2
from centers import model_center
from centers import model_center3
from centers import losses
from centers import aploss
from centers.center_utils import make_hm_regr, pool, pred2box, evaluate_keypoints, pred2centers,get_true_centers


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

if len(sys.argv) < 2:
        print("Использование: python script.py <путь_к_конфигу>")
        sys.exit(1)
config_path = sys.argv[1]

config = load_config(config_path)



test_id = config['model']['id']  #str(int(time.time()))


epochs = config['training']['num_epochs']   #100
batch_size = config['training']['batch_size'] 
num_workers = 2
oan_gamma = config['training']['oan_g'] 
oan_alpha = config['training']['oan_a'] 
resize_to = (config['training']['size'] , config['training']['size'] )
bb_pad = config['training']['bb_pad'] 
model_name = f'center_resize_{test_id}_h:{resize_to[0]}_w:{resize_to[1]}'

#optimazer
start_lr   = config['training']['learning_rate']
num_steps  = config['training']['num_steps']
gamma_coef = config['training']['gamma_coef']


print(f'epochs {epochs}')
print(f'batch_size {batch_size}')
print(f'num_workers {num_workers}')
print(f'oan_gamma {oan_gamma}')
print(f'oan_alpha {oan_alpha}')
print(f'start_lr {start_lr}')
print(f'num_steps {num_steps}')
print(f'gamma_coef {gamma_coef}')
print(f'resize_to {resize_to}')
print(f'bb_pad {bb_pad}')

print(f'id {test_id}')

training_type = config['model']['train_type']

weights_name = f'{model_name}'
# weights_name = f'{datetime.date.today().isoformat()}_retinanet_oan_vis+small_lr_lr{start_lr}_step{num_steps}_gamma{oan_gamma}_alpha{oan_alpha}'
path_to_save = f'/home/maantonov_1/VKR/weights/center/resize/test/{training_type}/{test_id}/'
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)
path_to_save = path_to_save + weights_name

print(f'path_to_save {path_to_save}')

input_size = resize_to[0] 
IN_SCALE = 1024//resize_to[0] 
MODEL_SCALE = 4

os.environ["WANDB_MODE"]="offline" 
wandb.init(project="VKR", entity="matvey_antonov", name = f"{weights_name}")



main_dir = config['data']['train_path']
images_dir_tarin = main_dir + 'images'


annot_path = config['data']['train_annot_path']
annotations_file_train = main_dir + annot_path


main_dir = config['data']['train_path']
images_dir_val = main_dir + 'images'
annot_path = config['data']['valid_annot_path']
annotations_file_val = main_dir + annot_path

main_dir = '/home/maantonov_1/VKR/data/main_data/test/'
images_dir_test = main_dir + 'images'
annotations_file_test = main_dir + 'true_annotations/annot.json'


dali_iterator_train = DALIGenericIterator(
    pipelines=[get_dali_pipeline_aug(images_dir = images_dir_tarin, annotations_file = annotations_file_train, resize_dims = resize_to, batch_size = batch_size, num_threads = num_workers)],
    output_map=['data', 'bboxe', 'bbox_shapes', 'img_id', 'horizontal_flip','vertical_flip', 'original_sizes'],#, 'angles'],
    reader_name='Reader',
    last_batch_policy=LastBatchPolicy.PARTIAL,
    auto_reset=True,
    dynamic_shape=True
)



dali_iterator_val = DALIGenericIterator(
    pipelines=[get_dali_pipeline(images_dir = images_dir_test, annotations_file = annotations_file_test, resize_dims = resize_to, batch_size = batch_size, num_threads = num_workers)],
    output_map=['data', 'bboxe', 'bbox_shapes', 'img_id', 'original_sizes'],
    reader_name='Reader',
    last_batch_policy=LastBatchPolicy.PARTIAL,
    auto_reset=True,
    dynamic_shape=True
)

dali_iterator_test = DALIGenericIterator(
    pipelines=[get_dali_pipeline(images_dir = images_dir_test, annotations_file = annotations_file_test, resize_dims = resize_to, batch_size = 1, num_threads = num_workers)],
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




if config['model']['model_type'] == '1':     
    model = model_center.resnet50(num_classes = 1, pretrained = False, inputs = 3)
    print(f'model type 1')
elif config['model']['model_type'] == '2':
    model = model_center2.resnet50(num_classes = 1, pretrained = False, inputs = 3)
    print(f'model type 2')
elif config['model']['model_type'] == '3':
    model = model_center3.resnet50(num_classes = 1, pretrained = False, inputs = 3)
    print(f'model type 3')

criterion = losses.centerloss2

if config['model']['pretrain_on_ImageNet']:
    weights = model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth', model_dir='.')
    # print(retinanet)
    print(model.load_state_dict(weights, strict=False))
    # print(retinanet)
    
    print(f'pretrain_on_ImageNet')

if config['training']['random_bb_pad']:
    print('random_bb_pad')

# retinanet = model_incpetion.resnetCustom(num_classes = 2, layers = [3, 10, 6, 3], inputs = 3)
# retinanet = torch.load(path_to_weights, map_location=device)


# criterion_type = config['training']['criterion']

# if criterion_type == 'focal':
#     criterion = losses.FocalLoss(alpha = oan_alpha, gamma = oan_gamma)
# elif criterion_type == 'aploss':
#     criterion = aploss.APLoss.apply
# print(criterion_type)





# retinanet.focalLoss = losses.FocalLoss(alpha = oan_alpha, gamma = oan_gamma)
# retinanet.aploss = aploss.APLoss.apply


optimizer = optim.AdamW(model.parameters(), lr = start_lr)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = num_steps, gamma=gamma_coef)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, verbose=True)

model.to(device)



def train_one_epoch(epoch_num, train_data_loader):
    torch.cuda.empty_cache()
    print("Train Epoch - {} Started".format(epoch_num), flush=True)
    st = time.time()
    
    model.train()
    
    epoch_loss = []

    for iter_num, data in enumerate(train_data_loader):
                
        optimizer.zero_grad()
        
        # img, annot = data['img'].cuda().float(), data['annot'].cuda().float()
        
        img = data[0]['data']
        bbox = data[0]['bboxe'].int()
        z = data[0]['img_id']
        h_flip = data[0]['horizontal_flip']
        v_flip = data[0]['vertical_flip']
        bb_shape = data[0]['bbox_shapes']
        original_sizes = data[0]['original_sizes']
        # angles = data[0]['angles']
        
        bb_pad_loc = bb_pad
        
        if config['training']['random_bb_pad']:
            bb_pad_loc = random.uniform(0, 0.5)
        
        bbox = resize_bb(bbox, original_sizes, bb_pad = bb_pad_loc, new_shape = resize_to)
        bbox = flip_bboxes(bbox, h_flip, v_flip, bb_shape, img_size=resize_to)
        # bbox = rotate_bboxes(bbox, angles, bb_shape, img_size=resize_to)
        
        hm_gt, regr_gt = make_hm_regr(bbox, bb_shape)
        
        # new_elements = torch.where(bbox[:, :, 3] == -1, -1, 1).unsqueeze(2)  
        # annot = torch.cat((bbox, new_elements), dim=2).to(device)
        
        
        
        
        hm, regr = model(img)
        preds = torch.cat((hm, regr), 1)
        
        loss, mask_loss = criterion(preds, hm_gt.to(device)) #size_average=False

        regr_loss = 0

        if bool(loss == 0):
            # print(
            # 'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | oan loss: {:1.5f} | Running loss: {:1.5f}'.format(
            #     epoch_num, iter_num, float(classification_loss), float(regression_loss), float(oan_loss), np.mean(epoch_loss)), flush=True)
            continue
                
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.1)
                
        optimizer.step()
        epoch_loss.append(float(loss))

        if iter_num % 10 == 0:
            print(
                'Epoch: {} | Iteration: {} | mask_loss: {:1.5f} | regr_loss: {:1.5f} | Running loss: {:1.5f}'.format(
                    epoch_num, iter_num, float(mask_loss), float(regr_loss),  np.mean(epoch_loss)), flush=True)
        
        del mask_loss
        del regr_loss
       
        
        
    # Update the learning rate
    if lr_scheduler is not None:
        lr_scheduler.step()
        
    et = time.time()
    print("\n Total Time - {}\n".format(int(et - st)))
    return np.mean(epoch_loss)
        
def valid_one_epoch(epoch_num, valid_data_loader):
    torch.cuda.empty_cache()
    print("Val Epoch - {} Started".format(epoch_num))
    st = time.time()
    model.train()

    epoch_loss = []

    for iter_num, data in enumerate(valid_data_loader):
                
        with torch.no_grad():
            
            img = data[0]['data']
            bbox = data[0]['bboxe'].int()
            z = data[0]['img_id']
            bb_shape = data[0]['bbox_shapes']
            original_sizes = data[0]['original_sizes']
            bbox = resize_bb(bbox, original_sizes, bb_pad = bb_pad, new_shape = resize_to)
        
            hm_gt, regr_gt = make_hm_regr(bbox, bb_shape)
        
            # new_elements = torch.where(bbox[:, :, 3] == -1, -1, 1).unsqueeze(2)  
            # annot = torch.cat((bbox, new_elements), dim=2).to(device)
            
            hm, regr = model(img)
            preds = torch.cat((hm, regr), 1)
            
            loss, mask_loss = criterion(preds, hm_gt.to(device)) #size_average=False

            regr_loss = 0

            epoch_loss.append(float(loss))

            if iter_num % 10 == 0:
                print(
                    'Epoch: {} | Iteration: {} | mask_loss: {:1.5f} | regr_loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(mask_loss), float(regr_loss),  np.mean(epoch_loss)), flush=True)
                
            # break
            
        
        del mask_loss
        del regr_loss
        
        
        
    et = time.time()
    print("\n Total Time - {}\n".format(int(et - st)))
    
    return np.mean(epoch_loss)
    
def get_metric_one_epoch(epoch_num, valid_data_loader, best_val, mode = 'val'):
    
    torch.cuda.empty_cache()
    
    print("GetMetric Epoch - {} Started".format(epoch_num))
    st = time.time()
    model.eval()

    epoch_loss = []
    prediction = []
    time_running = []
    
    gTP, gFP, gFN = 0,0,0

    for iter_num, data in enumerate(valid_data_loader):
                
        with torch.no_grad():
            
            img = data[0]['data']
            bbox = data[0]['bboxe']
            z = data[0]['img_id']
            bb_shape = data[0]['bbox_shapes'].cpu()
            original_sizes = data[0]['original_sizes']
            bbox = resize_bb(bbox, original_sizes, bb_pad = bb_pad, new_shape = resize_to).int()
            
            hm_gt, regr_gt = make_hm_regr(bbox, bb_shape)
        
            new_elements = torch.where(bbox[:, :, 3] == -1, -1, 1).unsqueeze(2)  
            annot = torch.cat((bbox, new_elements), dim=2).cpu()
            
            t = time.time()
            hm, regr = model(img)
            t1 = time.time()
            time_running.append(t1-t)
            
            preds = torch.cat((hm, regr), 1)
            
            loss, mask_loss = criterion(preds, hm_gt.to(device)) #size_average=False

            regr_loss = 0
            epoch_loss.append(float(loss))
            if iter_num % 10 == 0:
                print(
                    'Epoch: {} | Iteration: {} | mask_loss: {:1.5f} | regr_loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(mask_loss), float(regr_loss),  np.mean(epoch_loss)), flush=True)
        del mask_loss
        del regr_loss
            
    print(f'AVG time {np.mean(time_running)}')
    #     # pred = torch.cat((hm, regr), 1)
    #     # hm, regr = pred
    #     # for hm, regr in zip(hms, regrs):
    #     hm = hm.squeeze(0).squeeze(0).cpu().numpy()
    #     regr = regr.squeeze(0).cpu().numpy()

    #     hm = torch.sigmoid(torch.from_numpy(hm)).numpy()
    #     hm = pool(hm)

    #     first_scores, pred_center = pred2centers(hm, regr, thresh=0.5)
    #     pred_center = (pred_center[0] * MODEL_SCALE, pred_center[1] * MODEL_SCALE)
    #     centers, first_scores = get_true_centers(pred_center, first_scores, dist=20)

    #     active_annot = [bbox[i, :bb_shape[i, 0]] for i in range(bb_shape.size(0))]
    #     active_annot_tensor = torch.cat(active_annot, dim=0)
        
    #     if active_annot_tensor.shape[0] != 0:
    #         x = (active_annot_tensor[:,2] + active_annot_tensor[:,0])//2
    #         y = (active_annot_tensor[:,3] + active_annot_tensor[:,1])//2
    #         true_centers = np.column_stack((x, y))
    #     else:
    #         true_centers = []

        
    #     TP, FP, FN = evaluate_keypoints(true_centers, pred_center, radius=40)
    #     gTP += TP
    #     gFP += FP
    #     gFN += FN

        
    # precision = TP / (TP + FP) if TP + FP > 0 else 0
    # recall = TP / (TP + FN) if TP + FN > 0 else 0
    # f_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

            
    # print(
    #         'Epoch: {} | precision: {:1.5f} | recall: {:1.5f} | f_score: {:1.5f} | AVG time: {:1.5f}'.format(
    #             epoch_num, float(precision), float(recall), float(f_score), float(np.mean(time_running))), flush=True)
        


    et = time.time()
    print("\n Total Time - {}\n".format(int(et - st)))
    
    if best_val > np.mean(epoch_loss):
        best_val = np.mean(epoch_loss)
        torch.save(model, f"{path_to_save}.pt")
        print('SAVE PT')
    if epoch_num >= epochs - 1:
        torch.save(model, f"{path_to_save}_last.pt")
        
    return best_val, np.mean(epoch_loss)
    
best_val = 100
    
for epoch in range(epochs):
    st = time.time()

    mean_loss_train = train_one_epoch(epoch, dali_iterator_train)

    best_val, mean_loss_val = get_metric_one_epoch(epoch, dali_iterator_val, best_val)
    et = time.time()
    
    wandb.log({"epoch": epoch, "train_loss": float(mean_loss_train), "val_loss": float(mean_loss_val), "total_time":int(et - st)})
    
    