import os
import time
import numpy as np
import pandas as pd

import wandb
import torch
import torch.optim as optim
import torchvision.transforms as T
from torchvision.utils import make_grid 
from torch.utils.data import DataLoader

from retinanet.datasetLLAD import LLAD
from retinanet import model
from retinanet.dataloader import collater, ToTorch, Augmenter, Normalizer, UnNormalizer, AddDim
from retinanet import OAN



# os.environ["WANDB_MODE"]="offline" 
# wandb.init(project="VKR", entity="matvey_antonov")

DIR_TRAIN = '/home/maantonov_1/VKR/data/small_train/full_data/images'
train_df = pd.read_csv('/home/maantonov_1/VKR/data/small_train/train_small.csv')
valid_df = pd.read_csv('/home/maantonov_1/VKR/data/small_train/test_small.csv')


valid_dataset = LLAD(valid_df, DIR_TRAIN, mode = "valid", smart_crop = False, transforms = T.Compose([Normalizer(), ToTorch()]))

print(f'dataset Created', flush=True)



valid_data_loader = DataLoader(
    valid_dataset,
    batch_size = 16,
    shuffle = True,
    num_workers = 2,
    collate_fn = collater
)

for iter_num, data in enumerate(train_data_loader):
                
        
        img = data['annot']
        print(img.size())
        exit()