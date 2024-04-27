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
from retinanet import model_oan, model_incpetion, model_incpetion_new_fpn
from retinanet import losses
from retinanet import aploss
from retinanet.model_incpetion import PyramidFeatures, CustomPyramidFeatures, CustomPyramidFeaturesAT, CustomPyramidFeaturesAT2, CustomPyramidFeaturesR2, CustomPyramidFeaturesATR2, CustomPyramidFeaturesAT2_newLayer, CustomPyramidFeaturesAT2_newLayer_P2
from retinanet.center_utils import make_hm_regr, pool, pred2box, pred2centers, get_true_centers, calculate_accuracy_metrics

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
model_name = f'retinanet_resize_{test_id}_h:{resize_to[0]}_w:{resize_to[1]}'

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
path_to_save = f'/home/maantonov_1/VKR/weights/retinanet/resize/test/{training_type}/{test_id}/'
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)
path_to_save = path_to_save + weights_name

print(f'path_to_save {path_to_save}')


# path_to_weights = '/home/maantonov_1/VKR/weights/retinanet/visdrone/retinanet_oan_vis_lr0.0003_step_5_gamma:2_alpha:0.25_n26_m:0.03_f:0.04.pt'
path_to_weights = '/home/maantonov_1/VKR/weights/retinanet/resize/test/main/1713810939/retinanet_resize_1713810939_h:1312_w:1312_main.pt'

os.environ["WANDB_MODE"]="offline" 
wandb.init(project="VKR", entity="matvey_antonov", name = f"{weights_name}")


# main_dir = '/home/maantonov_1/VKR/data/crope_data/main/crop_train_1024/'
# main_dir = '/home/maantonov_1/VKR/data/crope_data/small/small_crop_train/'
# main_dir = '/home/maantonov_1/VKR/data/main_data/train/'
# main_dir = '/home/maantonov_1/VKR/data/small_train/train/'
# main_dir = '/home/maantonov_1/VKR/data/main_data/super_crop_1312/'



main_dir = config['data']['train_path']
images_dir_tarin = main_dir + 'images'
# annotations_file_train = main_dir + 'train_annot/annot.json'
# annotations_file_train = main_dir + 'more_sum_train_annot/annot.json'
# annotations_file_train = main_dir + 'true_train_annot/annot.json'
# annotations_file_train = main_dir + 'more_sum_less_empty_train_annot/annot.json'

annot_path = config['data']['train_annot_path']
annotations_file_train = main_dir + annot_path

# main_dir = '/home/maantonov_1/VKR/data/crope_data/main/crop_val_1024/'
# main_dir = '/home/maantonov_1/VKR/data/crope_data/small/small_crop_val/'
# main_dir = '/home/maantonov_1/VKR/data/main_data/train/'
# main_dir = '/home/maantonov_1/VKR/data/small_train/train/'
# main_dir = '/home/maantonov_1/VKR/data/main_data/super_crop_1312/'

main_dir = config['data']['train_path']
images_dir_val = main_dir + 'images'
# annotations_file_val = main_dir + 'val_annot/annot.json'
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
    pipelines=[get_dali_pipeline(images_dir = images_dir_val, annotations_file = annotations_file_val,resize_dims = resize_to, batch_size = batch_size, num_threads = num_workers)],
    output_map=['data', 'bboxe', 'bbox_shapes', 'img_id', 'original_sizes'],
    reader_name='Reader',
    last_batch_policy=LastBatchPolicy.PARTIAL,
    auto_reset=True,
    dynamic_shape=True
)

dali_iterator_test = DALIGenericIterator(
    pipelines=[get_dali_pipeline(images_dir = images_dir_test, annotations_file = annotations_file_test, resize_dims = resize_to, batch_size = batch_size, num_threads = num_workers)],
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


fpn_type = config['model']['fpn']


if fpn_type == 'PyramidFeatures':
    fpn = PyramidFeatures
    retinanet = model_incpetion.resnet50(num_classes = 2, pretrained = False, inputs = 3, fpn = fpn)
elif fpn_type == 'CustomPyramidFeatures':
    fpn = CustomPyramidFeatures
    retinanet = model_incpetion.resnet50(num_classes = 2, pretrained = False, inputs = 3, fpn = fpn)
elif fpn_type == 'CustomPyramidFeaturesAT':
    fpn = CustomPyramidFeaturesAT
    retinanet = model_incpetion.resnet50(num_classes = 2, pretrained = False, inputs = 3, fpn = fpn)
elif fpn_type == 'CustomPyramidFeaturesAT2':
    fpn = CustomPyramidFeaturesAT2
    retinanet = model_incpetion.resnet50(num_classes = 2, pretrained = False, inputs = 3, fpn = fpn)
elif fpn_type == 'CustomPyramidFeaturesR2':
    fpn = CustomPyramidFeaturesR2
    retinanet = model_incpetion.resnet50(num_classes = 2, pretrained = False, inputs = 3, fpn = fpn)
elif fpn_type == 'CustomPyramidFeaturesATR2':
    fpn = CustomPyramidFeaturesATR2
    retinanet = model_incpetion.resnet50(num_classes = 2, pretrained = False, inputs = 3, fpn = fpn)
elif fpn_type == 'CustomPyramidFeaturesAT2_newLayer':
    fpn = CustomPyramidFeaturesAT2_newLayer
    retinanet = model_incpetion.resnet50(num_classes = 2, pretrained = False, inputs = 3, fpn = fpn)
elif fpn_type == 'CustomPyramidFeaturesAT2_newLayer_P2':
    fpn = CustomPyramidFeaturesAT2_newLayer_P2
    retinanet = model_incpetion_new_fpn.resnet50(num_classes = 2, pretrained = False, inputs = 3, fpn = fpn)
    
print(f'FPN: {fpn_type}')



# retinanet = model_incpetion.resnetCustom(num_classes = 2, layers = [3, 10, 6, 3], inputs = 3)
# retinanet = torch.load(path_to_weights, map_location=device)


criterion_type = config['training']['criterion']

if criterion_type == 'focal':
    criterion = losses.FocalLoss(alpha = oan_alpha, gamma = oan_gamma)
elif criterion_type == 'aploss':
    criterion = aploss.APLoss.apply
print(criterion_type)





# retinanet.focalLoss = losses.FocalLoss(alpha = oan_alpha, gamma = oan_gamma)
# retinanet.aploss = aploss.APLoss.apply


optimizer = optim.AdamW(retinanet.parameters(), lr = start_lr)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = num_steps, gamma=gamma_coef)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, verbose=True)

retinanet.to(device)



def train_one_epoch(epoch_num, train_data_loader):
    torch.cuda.empty_cache()
    print("Train Epoch - {} Started".format(epoch_num), flush=True)
    st = time.time()
    
    retinanet.train()
    
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
        
        bbox = resize_bb(bbox, original_sizes, bb_pad = bb_pad, new_shape = resize_to)
        bbox = flip_bboxes(bbox, h_flip, v_flip, bb_shape, img_size=resize_to)
        # bbox = rotate_bboxes(bbox, angles, bb_shape, img_size=resize_to)
        
        new_elements = torch.where(bbox[:, :, 3] == -1, -1, 1).unsqueeze(2)  
        annot = torch.cat((bbox, new_elements), dim=2).to(device)
        
        
        # classification_loss, regression_loss = retinanet([img, annot])
        
        classification_loss, regression_loss = criterion(*retinanet([img, annot]))
                
        classification_loss = classification_loss.mean()
        regression_loss = regression_loss.mean()
        # class_loss_ap = class_loss_ap.mean()
        # reg_loss_ap = reg_loss_ap.mean()


        loss = classification_loss + regression_loss #+ 4 * oan_loss #+ class_loss_ap #+ reg_loss_ap
        oan_loss = 0

        if bool(loss == 0):
            # print(
            # 'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | oan loss: {:1.5f} | Running loss: {:1.5f}'.format(
            #     epoch_num, iter_num, float(classification_loss), float(regression_loss), float(oan_loss), np.mean(epoch_loss)), flush=True)
            continue
                
        loss.backward()

        torch.nn.utils.clip_grad_norm_(retinanet.parameters(), max_norm=0.5)
        
        torch.nn.utils.clip_grad_value_(retinanet.parameters(), clip_value=0.1)
                
        optimizer.step()
        epoch_loss.append(float(loss))

        if iter_num % 10 == 0:
            print(
                'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | oan loss: {:1.5f} | Running loss: {:1.5f}'.format(
                    epoch_num, iter_num, float(classification_loss), float(regression_loss), float(oan_loss), np.mean(epoch_loss)), flush=True)
        
        del classification_loss
        del regression_loss
        del oan_loss
        # break
       
        
        
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
    retinanet.train()

    epoch_loss = []

    for iter_num, data in enumerate(valid_data_loader):
                
        with torch.no_grad():
            
            img = data[0]['data']
            bbox = data[0]['bboxe'].int()
            z = data[0]['img_id']
            bb_shape = data[0]['bbox_shapes']
            original_sizes = data[0]['original_sizes']
            bbox = resize_bb(bbox, original_sizes, bb_pad = bb_pad, new_shape = resize_to)
        
            new_elements = torch.where(bbox[:, :, 3] == -1, -1, 1).unsqueeze(2)  
            annot = torch.cat((bbox, new_elements), dim=2).to(device)
            
            classification_loss, regression_loss = criterion(*retinanet([img, annot]))
                
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
                
            # classification_loss = classification_loss.mean()
            # regression_loss = regression_loss.mean()
            # class_loss_ap = class_loss_ap.mean()
            # reg_loss_ap = reg_loss_ap.mean()


            loss = classification_loss + regression_loss #+ 4 * oan_loss #+ class_loss_ap #+ reg_loss_ap
            oan_loss = 0

            epoch_loss.append(float(loss))

            if iter_num % 10 == 0:
                print(
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | oan loss: {:1.5f}  | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss), float(oan_loss), np.mean(epoch_loss)), flush=True)
        
        del classification_loss
        del regression_loss
        del oan_loss
        
        # break
        
    et = time.time()
    print("\n Total Time - {}\n".format(int(et - st)))
    
    return np.mean(epoch_loss)
    
def get_metric_one_epoch(epoch_num, valid_data_loader, best_val, last_val, metrics, mode = 'val'):
    
    torch.cuda.empty_cache()
    
    print("GetMetric Epoch - {} Started".format(epoch_num))
    st = time.time()
    retinanet.eval()

    epoch_loss = []
    prediction = []
    time_running = []
    
    precision_list = []
    recall_list = [] 
    f1_list = []

    for iter_num, data in enumerate(valid_data_loader):
                
        with torch.no_grad():
            
            img = data[0]['data']
            bbox = data[0]['bboxe']
            z = data[0]['img_id']
            bb_shape = data[0]['bbox_shapes'].cpu()
            original_sizes = data[0]['original_sizes']
            bbox = resize_bb(bbox, original_sizes, bb_pad = bb_pad, new_shape = resize_to).int()
            
            new_elements = torch.where(bbox[:, :, 3] == -1, -1, 1).unsqueeze(2)  
            annot = torch.cat((bbox, new_elements), dim=2).cpu()
            
            t = time.time()
            scores, labels, boxes = retinanet(img)
            t1 = time.time()
            
            pred_dict = {}
            
        time_running.append(t1-t)
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy() * 0
        boxes  = boxes.cpu().numpy() 
        pred_dict['scores'] = scores
        pred_dict['labels'] = labels
        pred_dict['boxes']  = boxes
        
        
        active_annot = [annot[i, :bb_shape[i, 0]] for i in range(bb_shape.size(0))]
        active_annot_tensor = torch.cat(active_annot, dim=0)
        # active_annot_np = active_annot_tensor.numpy()
        
        gt_dict = {}
        if annot.shape[0] != 0:
            gt_labels = active_annot_tensor[:,-1].numpy() * 0 
            gt_boxes  = active_annot_tensor[:,:-1].numpy()
        else:
            gt_labels = np.array([])
            gt_boxes  = np.array([])
        gt_dict['boxes'] = gt_boxes
        gt_dict['labels'] = gt_labels

        if active_annot_tensor.shape[0] != 0:
            x = (active_annot_tensor[:,2] + active_annot_tensor[:,0])//2
            y = (active_annot_tensor[:,3] + active_annot_tensor[:,1])//2
            true_centers = np.column_stack((x, y))
        else:
            true_centers = []
        
        if boxes.shape[0] != 0:
            x = (boxes[:,2] + boxes[:,0])//2
            y = (boxes[:,3] + boxes[:,1])//2
            pred_center = np.column_stack((x, y))
        else:
            pred_center = []
    
        precision, recall, f1 = calculate_accuracy_metrics(pred_center, true_centers, threshold = 50)
        
        precision_list.append(precision)
        recall_list.append(recall) 
        f1_list.append(f1)

        prediction.append((gt_dict, pred_dict))
        
        # break
        
    print(f'precision {np.mean(precision_list)}')
    print(f'recall {np.mean(recall_list)}')
    print(f'f1 {np.mean(f1_list)}')

    print('__________________')
        
    if mode == 'test':
        print(f'{datetime.date.today().isoformat()}')
        print(f'path_to_save {path_to_save}')
        print(f'AVG time: {np.mean(time_running)}')
        print()
        
        iou_thr_max = 0.6
        print(f'iou_thr_max: {iou_thr_max}')
        
        score_threshold = 0.05
        map_score, Fscore = evaluate(prediction, score_threshold = score_threshold, iou_thr_max=iou_thr_max)
        print(f'score_threshold: {score_threshold}')
        print(f'map_score: {map_score}')
        print(f'Fscore: {Fscore}')
        print('________________')

        for i_tresh in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            score_threshold = i_tresh
            map_score, Fscore = evaluate(prediction, score_threshold = score_threshold)
            print(f'score_threshold: {score_threshold}')
            print(f'map_score: {map_score}')
            print(f'Fscore: {Fscore}')
            print()
        return
        
 
    map_score, Fscore = evaluate(prediction)
            
    print(
            'Epoch: {} | map_score loss: {:1.5f} | Fscore loss: {:1.5f} | AVG time: {:1.5f}'.format(
                epoch_num, float(map_score), float(Fscore), float(np.mean(time_running))), flush=True)
        


    et = time.time()
    print("\n Total Time - {}\n".format(int(et - st)))
    
    if best_val > last_val:
        best_val = last_val
        torch.save(retinanet, f"{path_to_save}.pt")
        metrics = [np.mean(precision_list), np.mean(recall_list), np.mean(f1_list), epoch_num]
        print('SAVE PT')
    if epoch_num >= epochs - 1:
        torch.save(retinanet, f"{path_to_save}_last.pt")
        
    return map_score, Fscore, best_val, metrics
    
best_val = 100

last_metricsc = [0,0,0, 0]
    
for epoch in range(epochs):
    st = time.time()

    mean_loss_train = train_one_epoch(epoch, dali_iterator_train)
    
    mean_loss_val = valid_one_epoch(epoch, dali_iterator_val)

    map_score, Fscore, best_val, last_metricsc = get_metric_one_epoch(epoch, dali_iterator_val, best_val, mean_loss_val, last_metricsc)
    et = time.time()
    
    wandb.log({"epoch": epoch, "train_loss": float(mean_loss_train), "val_loss": float(mean_loss_val), 
                   "map_score": float(map_score), "Fscore": float(Fscore), "total_time":int(et - st)})
    
print(last_metricsc)
    
get_metric_one_epoch(0, dali_iterator_test, 0, 0, mode = 'test')