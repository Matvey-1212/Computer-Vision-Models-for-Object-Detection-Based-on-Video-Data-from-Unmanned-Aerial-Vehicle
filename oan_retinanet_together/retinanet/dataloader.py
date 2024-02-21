from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import pandas as pd
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


# class CocoDataset(Dataset):
#     """Coco dataset."""

#     def __init__(self, root_dir, set_name='train2017', transform=None):
#         """
#         Args:
#             root_dir (string): COCO directory.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         self.root_dir = root_dir
#         self.set_name = set_name
#         self.transform = transform

#         self.coco      = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
#         self.image_ids = self.coco.getImgIds()

#         self.load_classes()

#     def load_classes(self):
#         # load class names (name -> label)
#         categories = self.coco.loadCats(self.coco.getCatIds())
#         categories.sort(key=lambda x: x['id'])

#         self.classes             = {}
#         self.coco_labels         = {}
#         self.coco_labels_inverse = {}
#         for c in categories:
#             self.coco_labels[len(self.classes)] = c['id']
#             self.coco_labels_inverse[c['id']] = len(self.classes)
#             self.classes[c['name']] = len(self.classes)

#         # also load the reverse (label -> name)
#         self.labels = {}
#         for key, value in self.classes.items():
#             self.labels[value] = key

#     def __len__(self):
#         return len(self.image_ids)

#     def __getitem__(self, idx):

#         img = self.load_image(idx)
#         annot = self.load_annotations(idx)
#         sample = {'img': img, 'annot': annot}
#         if self.transform:
#             sample = self.transform(sample)

#         return sample

#     def load_image(self, image_index):
#         image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
#         path       = os.path.join(self.root_dir, 'images', self.set_name, image_info['file_name'])
#         img = skimage.io.imread(path)

#         if len(img.shape) == 2:
#             img = skimage.color.gray2rgb(img)

#         return img.astype(np.float32)/255.0

#     def load_annotations(self, image_index):
#         # get ground truth annotations
#         annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
#         annotations     = np.zeros((0, 5))

#         # some images appear to miss annotations (like image with id 257034)
#         if len(annotations_ids) == 0:
#             return annotations

#         # parse annotations
#         coco_annotations = self.coco.loadAnns(annotations_ids)
#         for idx, a in enumerate(coco_annotations):

#             # some annotations have basically no width / height, skip them
#             if a['bbox'][2] < 1 or a['bbox'][3] < 1:
#                 continue

#             annotation        = np.zeros((1, 5))
#             annotation[0, :4] = a['bbox']
#             annotation[0, 4]  = self.coco_label_to_label(a['category_id'])
#             annotations       = np.append(annotations, annotation, axis=0)

#         # transform from [x, y, w, h] to [x1, y1, x2, y2]
#         annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
#         annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

#         return annotations

#     def coco_label_to_label(self, coco_label):
#         return self.coco_labels_inverse[coco_label]


#     def label_to_coco_label(self, label):
#         return self.coco_labels[label]

#     def image_aspect_ratio(self, image_index):
#         image = self.coco.loadImgs(self.image_ids[image_index])[0]
#         return float(image['width']) / float(image['height'])

#     def num_classes(self):
#         return 80


class CSVDataset(Dataset):
    """CSV dataset."""

    def __init__(self, train_file, class_list, path,labels_format = 'min_max', transform=None, end_format='.jpg'):
        """
        Args:
            train_file (string): CSV file with training annotations
            annotations (string): CSV file with class list
            test_file (string, optional): CSV file with testing annotations
        """
        self.train_file = train_file
        self.class_list = class_list
        self.transform = transform
        self.data_path = path
        self.end_format = end_format

        # parse the provided class file
        try:
            self.classes = self.load_classes(pd.read_csv(self.class_list))
        except ValueError as e:
            raise(ValueError('invalid CSV class file: {}: {}'.format(self.class_list, e)))

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key
        
        # csv with img_path, x1, y1, x2, y2, class_name
        try:
            self.image_data = self._read_annotations(pd.read_csv(self.train_file), self.labels)
        except ValueError as e:
            raise(ValueError('invalid CSV annotations file: {}: {}'.format(self.train_file, e)))
        self.image_names = list(self.image_data.keys())

    def _parse(self, value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            raise_from(ValueError(fmt.format(e)), None)

    def _open_for_csv(self, path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')

    def load_classes(self, df):
        result = {}
        
        for index, row in df.iterrows():

            try:
                class_name, class_id = row['class_name'], int(row['class_id'])
            except ValueError:
                raise(ValueError('Error in parse class list'))
            

            if class_name in result:
                raise ValueError('index {}: duplicate class name: \'{}\''.format(index, class_name))
            result[class_name] = class_id
        return result

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def data_format_min_max_one_lab(self, x, y, w, h, width, height):
        x1 = int(x * width - w * width / 2)
        y1 = int(y * height - h * height / 2)
        x2 = int(x1 + w * width)
        y2 = int(y1 + h * height)

        return int(x1), int(y1), int(x2), int(y2)

    def load_image(self, image_index):
        img = skimage.io.imread(self.data_path +'/' +self.image_names[image_index]+self.end_format)

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32)/255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotation_list = self.image_data[self.image_names[image_index]]
        annotations     = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotation_list) == 0:
            return annotations

        # parse annotations
        for idx, a in enumerate(annotation_list):
            # some annotations have basically no width / height, skip them
            x1 = a['x1']
            x2 = a['x2']
            y1 = a['y1']
            y2 = a['y2']

            if (x2-x1) < 1 or (y2-y1) < 1:
                continue

            annotation        = np.zeros((1, 5))
            
            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2

            annotation[0, 4]  = a['class']
            annotations       = np.append(annotations, annotation, axis=0)

        return annotations

    def _read_annotations(self, df, classes):
        result = {}
        for index, row in df.iterrows():

            try:
                x1, y1, x2, y2 = row['x'], row['y'], row['w'], row['h']
                img_file, class_name = str(int(row['id'])),  int(row['class'])
                width, height = int(row['width']), int(row['height'])
            except ValueError:
                raise_from(ValueError('line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(index)), None)

            if img_file not in result:
                result[img_file] = []


            if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
                continue
            x1, y1, x2, y2 = self.data_format_min_max_one_lab(x1, y1, x2, y2, width, height)


            if x2 <= x1:
                raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(index, x2, x1))
            if y2 <= y1:
                raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(index, y2, y1))


            if class_name not in classes:
                raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(index, class_name, classes))

            result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
        return result

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def num_classes(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)


def collater(data):

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

    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}

class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, min_side=608, max_side=1024):
        image, annots = sample['img'], sample['annot']

        rows, cols, cns = image.shape

#         smallest_side = min(rows, cols)

#         # rescale the image so the smallest side is min_side
#         scale = min_side / smallest_side

#         # check if the largest side is now greater than max_side, which can happen
#         # when images have a large aspect ratio
#         largest_side = max(rows, cols)

#         if largest_side * scale > max_side:
#             scale = max_side / largest_side

#         # resize the image with the computed scale
#         image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
#         rows, cols, cns = image.shape

        pad_w = 64 - rows%64
        pad_h = 64 - cols%64

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': 1}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):

        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()
            
            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):

        image, annots = sample['img'], sample['annot']

        return {'img':((image.astype(np.float32)-self.mean)/self.std), 'annot': annots}
#         return {'img':(image//255), 'annot': annots}

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
