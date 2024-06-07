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
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

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


# model = create_model(num_classes = 2)

# import torchvision
# from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights
# from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# # model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
# #                                                                        num_classes = 2,
# #                                                                        trainable_backbone_layers = 6,
# #                                                                        progress = False)
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# in_features = model.roi_heads.box_predictor.cls_score.in_features
# print(in_features)
# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)


# model.eval()
# x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
# predictions = model(x)

torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='COCO_V1')