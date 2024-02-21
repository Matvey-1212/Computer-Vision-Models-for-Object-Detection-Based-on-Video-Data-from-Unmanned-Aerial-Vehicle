import os
import time
import numpy as np
import pandas as pd

# import wandb
import torch
import torch.optim as optim
import torchvision.transforms as T
from torchvision.utils import make_grid 
from torch.utils.data import DataLoader

from retinanet.datasetLLAD import LLAD
from retinanet import model
from retinanet.dataloader import collater, ToTorch, Augmenter, Normalizer, UnNormalizer


# os.environ["WANDB_MODE"]="offline" 
# wandb.init(project="VKR", entity="matvey_antonov")

DIR_TRAIN = '/home/maantonov_1/VKR/data/small_train/full_data/images'
train_df = pd.read_csv('/home/maantonov_1/VKR/data/small_train/train_small.csv')
valid_df = pd.read_csv('/home/maantonov_1/VKR/data/small_train/test_small.csv')

# train_dataset = LLAD(train_df, DIR_TRAIN, mode = "train", transforms = T.Compose([Augmenter(),  ToTorch()]))
valid_dataset = LLAD(valid_df, DIR_TRAIN, mode = "valid", transforms = T.Compose([ ToTorch()]))

print(f'dataset Created', flush=True)

# DataLoaders
# train_data_loader = DataLoader(
#     train_dataset,
#     batch_size = 32,
#     shuffle = True,
#     num_workers = 4,
#     collate_fn = collater
# )

valid_data_loader = DataLoader(
    valid_dataset,
    batch_size = 32,
    shuffle = True,
    num_workers = 4,
    collate_fn = collater
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()
print('CUDA available: {}'.format(torch.cuda.is_available()), flush=True)
print(f'Crreating retinanet ===>', flush=True)

# retinanet = model.resnet50(num_classes = 1, pretrained = False)

# optimizer = torch.optim.Adam(retinanet.parameters(), lr = 0.0001)

# # Learning Rate Scheduler
# #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma=0.5)

# retinanet.to(device)

#No of epochs
epochs = 1
model_path = '/home/maantonov_1/VKR/weights/retinanet/retinanet_gwd.pt'
retinanet=torch.load(model_path)

use_gpu = True

if use_gpu:
    if torch.cuda.is_available():
        retinanet = retinanet.cuda()

if torch.cuda.is_available():
    #retinanet.load_state_dict(torch.load(parser.model_path))
    retinanet = torch.nn.DataParallel(retinanet).cuda()
else:
    retinanet.load_state_dict(torch.load(model_path))
    retinanet = torch.nn.DataParallel(retinanet)

retinanet.training = False
retinanet.eval()
retinanet.module.freeze_bn()
    
def valid_one_epoch(epoch_num, valid_data_loader):
    
    print("Epoch - {} Started".format(epoch_num))
    st = time.time()
    
    epoch_loss = []

    for iter_num, data in enumerate(valid_data_loader):
                
        with torch.no_grad():
            
            scores, labels, boxes = retinanet(data['img'].cuda().float())
            print(f'scores {scores}')
            print(f'labels {labels}')
            print(f'boxes {boxes}')
            # Forward
            # classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot'].cuda().float()])

            # # Calculating Loss
            # classification_loss = classification_loss.mean()
            # regression_loss = regression_loss.mean()
            # loss = classification_loss + regression_loss

            # #Epoch Loss
            # epoch_loss.append(float(loss))

            # print(
            #     'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
            #         epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(epoch_loss)))

            # del classification_loss
            # del regression_loss
        break
        
    et = time.time()
    print("\n Total Time - {}\n".format(int(et - st)))
    

    # torch.save(retinanet, "retinanet_gwd.pt")
    return np.mean(epoch_loss)
    
    
for epoch in range(epochs):
    st = time.time()
    
    mean_loss_val = valid_one_epoch(epoch, valid_data_loader)
    et = time.time()
    
    # wandb.log({"epoch": epoch, "train_loss": float(mean_loss_train), 
    #                "val_loss": float(mean_loss_val),  "total_time":int(et - st)})