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
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

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



epochs = 30
batch_size = 14
num_workers = 2
resize_to = (1024, 1024)
bb_pad = 0.1
model_name = f'faster_resize_h:{resize_to[0]}_w:{resize_to[1]}'

#optimazer
start_lr   = 0.0003
num_steps  = 10
gamma_coef = 0.5

print(f'epochs {epochs}')
print(f'batch_size {batch_size}')
print(f'num_workers {num_workers}')

print(f'start_lr {start_lr}')
print(f'num_steps {num_steps}')
print(f'gamma_coef {gamma_coef}')
print(f'resize_to {resize_to}')
print(f'bb_pad {bb_pad}')

weights_name = f'{datetime.date.today().isoformat()}_{model_name}_small+main_lr:{start_lr}_step:{num_steps}'
path_to_save = f'/home/maantonov_1/VKR/weights/faster/resize/main/{datetime.date.today().isoformat()}/'
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)
path_to_save = path_to_save + weights_name

print(f'path_to_save {path_to_save}')

# path_to_weights = '/home/maantonov_1/VKR/weights/faster/small/2024-03-24/2024-03-24_faster_main_lr:0.0003_step:5_n28_m:0.16_f:0.25_val:2.1593.pt'
path_to_weights = '/home/maantonov_1/VKR/weights/faster/resize/small/2024-03-24/2024-03-24_faster_resize_h:1024_w:1024_main_lr:0.0003_step:10_n29_m:0.29_f:0.19_val:0.8341.pt'

os.environ["WANDB_MODE"]="offline" 
wandb.init(project="VKR", entity="matvey_antonov", name = f"{weights_name}")


# main_dir = '/home/maantonov_1/VKR/data/crope_data/main/crop_train_1024/'
# main_dir = '/home/maantonov_1/VKR/data/crope_data/small/small_crop_train/'
# main_dir = '/home/maantonov_1/VKR/data/small_train/train/'
main_dir = '/home/maantonov_1/VKR/data/main_data/train/'
images_dir_tarin = main_dir + 'images'
# annotations_file_train = main_dir + 'train_annot/annot.json'
annotations_file_train = main_dir + 'more_sum_train_annot/annot.json'

# main_dir = '/home/maantonov_1/VKR/data/crope_data/main/crop_val_1024/'
# main_dir = '/home/maantonov_1/VKR/data/crope_data/small/small_crop_val/'
# main_dir = '/home/maantonov_1/VKR/data/small_train/train/'
main_dir = '/home/maantonov_1/VKR/data/main_data/train/'
images_dir_val = main_dir + 'images'
annotations_file_val = main_dir + 'val_annot/annot.json'

main_dir = '/home/maantonov_1/VKR/data/main_data/test/'
images_dir_test = main_dir + 'images'
annotations_file_test = main_dir + 'annotations/annot.json'


dali_iterator_train = DALIGenericIterator(
    pipelines=[get_dali_pipeline_aug(images_dir = images_dir_tarin, annotations_file = annotations_file_train,resize_dims = resize_to, batch_size = batch_size, num_threads = num_workers)],
    output_map=['data', 'bboxe', 'bbox_shapes', 'img_id', 'horizontal_flip','vertical_flip', 'original_sizes'],
    reader_name='Reader',
    last_batch_policy=LastBatchPolicy.PARTIAL,
    auto_reset=True,
    dynamic_shape=True
)

dali_iterator_val = DALIGenericIterator(
    pipelines=[get_dali_pipeline(images_dir = images_dir_val, annotations_file = annotations_file_val, resize_dims = resize_to,batch_size = batch_size, num_threads = num_workers)],
    output_map=['data', 'bboxe', 'bbox_shapes', 'img_id','original_sizes'],
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

# def create_model(num_classes):
#     backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2).features

#     backbone.out_channels = 960  

#     anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))

#     roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

#     model = FasterRCNN(backbone,
#                        num_classes=num_classes,
#                        rpn_anchor_generator=anchor_generator,
#                        box_roi_pool=roi_pooler)

#     return model

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



# model = create_model(num_classes = 2)
# model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(#weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
#                                                                        num_classes = 2,
#                                                                        trainable_backbone_layers = 6,
#                                                                        progress = False)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, trainable_backbone_layers = 5)
in_features = model.roi_heads.box_predictor.cls_score.in_features
# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

# model = torch.load(path_to_weights, map_location=device)


optimizer = optim.Adam(model.parameters(), lr = start_lr)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = num_steps, gamma=gamma_coef)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, verbose=True)

model.to(device)

def dict_list_to_dict(list_of_dicts):
    concatenated_dict = {}
    for key in list_of_dicts[0]:
        concatenated_dict[key] = torch.cat([d[key] for d in list_of_dicts], dim=0)
    return concatenated_dict

def train_one_epoch(epoch_num, train_data_loader):
    
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
        bbox = resize_bb(bbox, original_sizes, bb_pad = bb_pad, new_shape = resize_to)
        bbox = flip_bboxes(bbox, h_flip, v_flip, bb_shape)

        
        img, annot = transform_annotations(img, bbox)
        
        try:
            loss_dict = model(img, annot)
        except:
            print(z)
            continue
        losses = sum(loss for loss in loss_dict.values())
        

        if bool(losses == 0):
            print(
            'Epoch: {} | Iteration: {} |  loss: {:1.5f} | Running loss: {:1.5f}'.format(
                epoch_num, iter_num, float(losses), np.mean(epoch_loss)), flush=True)
            continue
                
        losses.backward()

                
        optimizer.step()
        epoch_loss.append(float(losses))

            
        print(
            'Epoch: {} | Iteration: {} |  loss: {:1.5f} | Running loss: {:1.5f}'.format(
                epoch_num, iter_num, float(losses), np.mean(epoch_loss)), flush=True)
        
        
    # Update the learning rate
    if lr_scheduler is not None:
        lr_scheduler.step()
        
    et = time.time()
    print("\n Total Time - {}\n".format(int(et - st)))
    return np.mean(epoch_loss)
        
def valid_one_epoch(epoch_num, valid_data_loader):
    
    print("Val Epoch - {} Started".format(epoch_num))
    st = time.time()
    model.train()

    epoch_loss = []

    for iter_num, data in enumerate(valid_data_loader):
                
        with torch.no_grad():
            
            img = data[0]['data']
            bbox = data[0]['bboxe'].int()
            z = data[0]['img_id']
            
            original_sizes = data[0]['original_sizes']
            bbox = resize_bb(bbox, original_sizes, bb_pad = bb_pad, new_shape = resize_to)
            
            img, annot = transform_annotations(img, bbox)
            try:
                loss_dict = model(img, annot)
            except:
                print(z)
                continue
            losses = sum(loss for loss in loss_dict.values())

            epoch_loss.append(float(losses))

            print(
            'Epoch: {} | Iteration: {} |  loss: {:1.5f} | Running loss: {:1.5f}'.format(
                epoch_num, iter_num, float(losses), np.mean(epoch_loss)), flush=True)
        

        
    last_val = np.mean(epoch_loss)
        
    et = time.time()
    print("\n Total Time - {}\n".format(int(et - st)))
    
    return np.mean(epoch_loss), last_val
    
def get_metric_one_epoch(epoch_num, valid_data_loader, best_val, last_val, mode = 'val'):
    
    print("GetMetric Epoch - {} Started".format(epoch_num))
    st = time.time()
    model.eval()

    epoch_loss = []
    prediction = []
    time_running = []

    for iter_num, data in enumerate(valid_data_loader):
                
        with torch.no_grad():
            
            img = data[0]['data']
            bbox = data[0]['bboxe'].int()
            z = data[0]['img_id']
            bb_shape = data[0]['bbox_shapes'].cpu()
            original_sizes = data[0]['original_sizes']
            bbox = resize_bb(bbox, original_sizes, bb_pad = bb_pad, new_shape = resize_to)

            
            img, annot = transform_annotations(img, bbox)
            try:
                t = time.time()
                pred_dict = model(img)
                t1 = time.time()
                time_running.append(t1-t)
            except:
                print(z)
                continue
        
        annot, pred_dict = dict_list_to_dict(annot), dict_list_to_dict(pred_dict)
            
        pred_dict['labels'] = pred_dict['labels'] * 0
        annot['labels'] = annot['labels'] * 0
            
        to_numpy(annot)
        to_numpy(pred_dict)

        prediction.append((annot, pred_dict))
        
    if mode == 'test':
        print(f'{datetime.date.today().isoformat()}')
        print(f'path_to_save {path_to_save}')
        print(f'AVG time: {np.mean(time_running)}')
        print()

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
            'Epoch: {} | map_score loss: {:1.5f} | Fscore loss: {:1.5f} '.format(
                epoch_num, float(map_score), float(Fscore)), flush=True)
        

        
    et = time.time()
    print("\n Total Time - {}\n".format(int(et - st)))
    
    if best_val > last_val:
        best_val = last_val
        torch.save(model, f"{path_to_save}.pt")
    elif epoch_num >= epochs - 1:
        torch.save(model, f"{path_to_save}_n{epoch_num}_m:{map_score:0.2f}_f:{Fscore:0.2f}_val:{last_val:0.4f}_last.pt")
        
    return map_score, Fscore
    
best_val = 100
last_val = 0
    
for epoch in range(epochs):
    st = time.time()

    mean_loss_train = train_one_epoch(epoch, dali_iterator_train)
    
    mean_loss_val, last_val = valid_one_epoch(epoch, dali_iterator_val)
    
    map_score, Fscore = get_metric_one_epoch(epoch, dali_iterator_val, best_val, last_val)
    et = time.time()
    
    wandb.log({"epoch": epoch, "train_loss": float(mean_loss_train), "val_loss": float(mean_loss_val), 
                   "map_score": float(map_score), "Fscore": float(Fscore), "total_time":int(et - st)})
    
    
get_metric_one_epoch(0, dali_iterator_test, 0, 0, mode='test')