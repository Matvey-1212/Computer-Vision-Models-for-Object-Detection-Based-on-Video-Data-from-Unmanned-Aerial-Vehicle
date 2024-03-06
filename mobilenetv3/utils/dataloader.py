from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import csv

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

# from pycocotools.coco import COCO

import skimage.io
import skimage.transform
import skimage.color
import skimage

from PIL import Image


def mosaic(img):
    a, b, c, d = img.shape
    
    mosaic_img = torch.zeros(1, b * 2, c * 2, d)
    
    mosaic_img[0, :640, :640, :] = img[0]
    mosaic_img[0, 640:, :640, :] = img[1]
    mosaic_img[0, :640, 640:, :] = img[2]
    mosaic_img[0, 640:, 640:, :] = img[3]
    return mosaic_img

def collater_mosaic(data):

    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]
    masks = [s['mask'] for s in data]

    
        
    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)
    padded_mask = torch.zeros(batch_size, max_width, max_height, 1)

    for i in range(batch_size):
        img = imgs[i]
        mask = masks[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img
        padded_mask[i, :int(mask.shape[0]), :int(mask.shape[1]), :] = mask.unsqueeze(-1)

    max_num_annots = max(annot.shape[0] for annot in annots)
    
    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                #print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1
        
    mosaic_imgs = torch.zeros(batch_size//4, max_width*2, max_height*2, 3)
    mosaic_mask = torch.zeros(batch_size//4, max_width*2, max_height*2, 1)
    
    for i in range(batch_size//4): 
        mosaic_imgs[i] = mosaic(padded_imgs[i*4:4*i+4])
        mosaic_mask[i] = mosaic(padded_mask[i*4:4*i+4])
        
    mosaic_imgs = mosaic_imgs.permute(0, 3, 1, 2)
    mosaic_mask = mosaic_mask.permute(0, 3, 1, 2)
    
    
    
    # print({'img': padded_imgs, 'annot': annot_padded, 'mask': padded_mask, 'scale': scales})

    return {'img': mosaic_imgs, 'annot': annot_padded, 'mask': mosaic_mask, 'scale': scales}

def collater_mask(data):

    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]
    masks = [s['mask'] for s in data]

    
        
    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)
    
    
    mask_widths = [int(s.shape[0]) for s in masks]
    mask_heights = [int(s.shape[1]) for s in masks]

    max_mask_width = np.array(mask_widths).max()
    max_mask_height = np.array(mask_heights).max()
    
    padded_mask = torch.zeros(batch_size, max_mask_width, max_mask_height, 1)

    for i in range(batch_size):
        img = imgs[i]
        mask = masks[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img
        padded_mask[i, :int(mask.shape[0]), :int(mask.shape[1]), :] = mask.unsqueeze(-1)

    max_num_annots = max(annot.shape[0] for annot in annots)
    
    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                #print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1
        
        
    padded_imgs = padded_imgs.permute(0, 3, 1, 2)
    padded_mask = padded_mask.permute(0, 3, 1, 2)
    

    return {'img': padded_imgs, 'annot': annot_padded, 'mask': padded_mask, 'scale': scales}

def collater_annot(data):

    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]


    
        
    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)


    for i in range(batch_size):
        img = imgs[i]

        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img


    max_num_annots = max(annot.shape[0] for annot in annots)
    
    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                #print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1
        
        
    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    
    # print({'img': padded_imgs, 'annot': annot_padded, 'mask': padded_mask, 'scale': scales})

    return {'img': padded_imgs, 'annot': annot_padded,  'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, min_side=608, max_side=1024):
        image, annots = sample['img'], sample['annot']

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
        rows, cols, cns = image.shape

        pad_w = 64 - rows%64
        pad_h = 64 - cols%64

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)
        
        # annots[:, :4] *= scale
        
        sample['img']   = torch.from_numpy(new_image)
        sample['annot'] = torch.from_numpy(annots)
        sample['scale'] = scale
        
        
        if 'mask' in sample:
            mask = sample['mask']
            rows, cols, cns = new_image.shape
            mask = skimage.transform.resize(mask, (int(round(rows//64)), int(round((cols//64)))))
            sample['mask'] = torch.from_numpy(mask)
            

        return sample
    
class ToTorch(object):
    def __call__(self, sample):
        image = sample['img']
        annots = sample['annot']
        
        # if 'annot' in sample:
        #     annots = sample['annot']
        #     return {'img': torch.from_numpy(image.copy()), 'annot': torch.from_numpy(annots.copy()), 'scale': 1}
        
        # if 'mask' in sample:
        #     mask = sample['mask']
        #     return {'img': torch.from_numpy(image.copy()), 'mask': torch.from_numpy(mask.copy()), 'annot': torch.from_numpy(annots.copy()), 'scale': 1}
        
        sample['img'] = torch.from_numpy(image.copy())
        sample['annot'] = torch.from_numpy(annots.copy())
        sample['scale'] = 1
        if 'mask' in sample:
            mask = sample['mask']
            sample['mask'] = torch.from_numpy(mask.copy())
        
        return sample #{'img': torch.from_numpy(image.copy()), 'annot': torch.from_numpy(annots.copy()),'scale': 1}
    
class AddDim(object):
    def __call__(self, sample):
        image = sample['img']
        print(f'!!!!!! {image.size()}')
        a, b, _ = image.size()
        
        # Создаем слой нулей
        zeros = torch.zeros(a, b, 1)

        # Конкатенируем исходный тензор с нулевым слоем вдоль второй оси (y)
        expanded_tensor = torch.cat([image, zeros], dim=2)
        
        sample['img'] = expanded_tensor
        
        return sample


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample, flip_x=0.5, flip_y=0.5):
        
        image, annots = sample['img'], sample['annot']
        
        if 'mask' in sample:
            mask = sample['mask']
            augmented = self.transform(image=image) #, mask=mask)
            image = augmented['image']
            # mask = augmented['mask']
        else:
            augmented = self.transform(image=image)
            image = augmented['image']
     
        rows, cols, channels = image.shape
        
        if np.random.rand() < flip_x:
            image = image[:, ::-1, :]
            
            if 'mask' in sample:
                mask = mask[:, ::-1]
            
            if annots.shape[0] != 0:
                x1 = annots[:, 0].copy()
                x2 = annots[:, 2].copy()
                
                x_tmp = x1.copy()

                annots[:, 0] = cols - x2
                annots[:, 2] = cols - x_tmp
            
        if np.random.rand() < flip_y:
            image = image[::-1, :, :]
            
            if 'mask' in sample:
                mask = mask[::-1, :]
            
            if annots.shape[0] != 0:
                y1 = annots[:, 1].copy()
                y2 = annots[:, 3].copy()
                
                y_tmp = y1.copy()

                annots[:, 1] = rows - y2
                annots[:, 3] = rows - y_tmp

        if 'mask' in sample:
            return {'img': image, 'mask': mask, 'annot': annots}
        
        return {'img': image, 'annot': annots}



class Normalizer(object):

    def __init__(self):
        # self.mean = np.array([[[0.501, 0.508, 0.497]]])
        # self.std = np.array([[[0.168, 0.174, 0.183]]])
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])


    def __call__(self, sample):

        image, annots = sample['img'], sample['annot']
        
        
        # if 'mask' in sample:
        #     mask = sample['mask']
            # return {'img':((image.astype(np.float32)-self.mean)/self.std),'mask': mask, 'annot': annots}
            
        sample['img'] = ((image.astype(np.float32)-self.mean)/self.std)

        return sample

class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]
