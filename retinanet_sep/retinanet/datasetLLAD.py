import random
import time
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import  Dataset
from scipy.ndimage import gaussian_filter

class LLAD(Dataset):

    def __init__(self, dataframe, image_dir, mode = "train", data_format='min_max', gaus = False, class_mask = False,  transforms = None, smart_crop = False, new_shape = (2048, 2048)):
        
        super().__init__()
        self.image_ids = dataframe['id']#.unique()
        self.image_lab = dataframe[['x', 'y', 'w', 'h','class','width','height']]
        self.df = dataframe
        self.image_dir = image_dir
        self.mode = mode
        self.transforms = transforms
        self.smart_crop = smart_crop
        self.new_shape = new_shape
        self.data_format = data_format
        self.gaus = gaus
        self.class_mask = class_mask
    
    def gaus_mask(self, img, annot):
        height, width, _ = img.shape
        
        target_map = np.zeros((int(height), int(width)), dtype = np.int64)
        
        sigma = 20
        is_gaus = False
        for coord in annot:
            x1 = coord[0]
            y1 = coord[1]
            x2 = coord[2]
            y2 = coord[3]
            x = int((x1 + x2)/2)
            y = int((y1 + y2)/2)
            
            target_map[y, x] = 10
            sigma = min(min(x2-x1, y2-y1), sigma)
            is_gaus = True
            
        if is_gaus:
            kernel_size = int(6*sigma + 1)  # Размер ядра должен быть нечетным
            if kernel_size % 2 == 0:
                kernel_size += 1
            blurred_image = cv2.GaussianBlur(target_map, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)
            return blurred_image
        else:
            return target_map
        
    def create_class_mask(self, img, annot):
        height, width, _ = img.shape
        
        target_map = np.zeros((int(height), int(width)))#, 2
        # target_map[:,:,0] = 1
        
        for coord in annot:
            x1 = int(coord[0])
            y1 = int(coord[1])
            x2 = int(coord[2])
            y2 = int(coord[3])
            
            
            target_map[y1:y2, x1:x2] = 1
            # target_map[y1:y2, x1:x2,0] = 0
            
        
        return target_map
        
        
    def random_crop_with_bbox(self, image, current_bbox, bboxes, window_size, teshold = 0.7, format = 'min_max'):
        """
        Выполняет рандомный кроп изображения, гарантируя, что в окне будет хотя бы один bbox.

        :param image: Изображение в виде NumPy array.
        :param bboxes: Список ограничивающих прямоугольников в формате [x_min, y_min, x_max, y_max].
        :param window_size: Размер окна в виде (высота, ширина).

        :return: Обрезанное изображение и список bbox, которые попали в окно.
        """
        img_height, img_width = image.shape[:2]
        crop_h, crop_w = window_size
        
        if len(bboxes) == 0:
            x_start = random.randint(0, img_width - crop_w)
            y_start = random.randint(0, img_height - crop_h)
            cropped_image = image[y_start:y_start + crop_h, x_start:x_start + crop_w]
            return cropped_image, np.array([])

        selected_bbox = current_bbox

        x_min, y_min, x_max, y_max, _ = selected_bbox

        x_start_max = min(img_width - crop_w, x_min)
        x_start_min = max(0, x_max - crop_w)
        y_start_max = min(img_height - crop_h, y_min)
        y_start_min = max(0, y_max - crop_h)   
        
        x_start = random.randint(x_start_min, x_start_max)
        y_start = random.randint(y_start_min, y_start_max)

        cropped_image = image[y_start:y_start + crop_h, x_start:x_start + crop_w]
        
        new_bboxes = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max, obj_class = bbox
            new_x_min = max(0, x_min - x_start)
            new_y_min = max(0, y_min - y_start)
            new_x_max = min(max(0, x_max - x_start), crop_w)
            new_y_max = min(max(0, y_max - y_start), crop_h)

            if new_x_min < crop_w and new_y_min < crop_h and 0 < new_x_max and 0 < new_y_max:
                if (new_x_max - new_x_min) * (new_y_max - new_y_min) / ((x_max - x_min) * (y_max - y_min)) >= teshold:
                    if format == 'min_max':
                        new_bboxes.append([new_x_min, new_y_min, new_x_max, new_y_max, obj_class])
                    elif format == 'yolo':
                        new_bboxes.append(self.min_max_to_yolo([new_x_min, new_y_min, new_x_max, new_y_max, obj_class], window_size))
        # print(np.array(new_bboxes))
        return cropped_image, np.array(new_bboxes)
    
    def min_max_to_yolo(self, label, window_size):
        img_height, img_width = window_size

        x_min = label[0]
        y_min = label[1]
        x_max = label[2]
        y_max = label[3]
        label[0] = 0
        label[1] = (x_max + x_min)/2 / img_width
        label[2] = (y_max + y_min)/2 / img_height
        label[3] = (x_max - x_min) / img_width
        label[4] = (y_max - y_min) / img_height
        return label
    
    def data_format_min_max(self, values, width, height):
        if np.sum(np.isnan(values)) > 0:
            return np.array([])
        
            
        boxes_orig = np.zeros((values.shape[0], 5))
        boxes_orig[:, 0:4] = values
        boxes_orig[:, 4] = boxes_orig[:, 4] * 0
        
        boxes = np.zeros((values.shape[0], 5))
        boxes[:, 0:4] = values
        boxes[:, 0] = (boxes[:, 0] * width - boxes[:, 2] * width / 2).astype(int)
        boxes[:, 1] = (boxes[:, 1] * height - boxes[:, 3] * height / 2).astype(int)
        boxes[:, 2] = (boxes[:, 0] + boxes[:, 2] * width).astype(int)
        boxes[:, 3] = (boxes[:, 1] + boxes[:, 3] * height).astype(int)
        boxes[:, 4] = 0

        return boxes
    
    def data_format_min_max_one_lab(self, values):
        x = values[0]
        y = values[1]
        w = values[2]
        h = values[3]
        width = values[5]
        height = values[6]
        
        if np.sum(np.isnan(values)) > 0:
            return np.array([])
        
        boxes = np.zeros(5)
        try:
            boxes[0] = int(x * width - w * width / 2)
            boxes[1] = int(y * height - h * height / 2)
            boxes[2] = int(boxes[0] + w * width)
            boxes[3] = int(boxes[1] + h * height)
            boxes[4] = 0
        except:
            print(values)
            exit()
        return boxes
    
    

    def __getitem__(self, index: int):

        # Retriving image id and records from df
        image_id = self.image_ids[index]
        image_lab = self.image_lab.iloc[index].values
        records = self.df[self.df['id'] == image_id]

        # t = time.time()
        
        # Loading Image
        image = cv2.imread(f'{self.image_dir}/{image_id}.JPG', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        
        # print(f'{index} load {time.time() - t}')
        # t = time.time()

        # If mode is set to train, then only we create targets
        if self.mode == "train" or self.mode == "valid":
            
            width = records['width'].values.astype(int)
            height = records['height'].values.astype(int)
            
            # if records[['x', 'y', 'w', 'h']].values.shape[0] == 1:
                # if records[['x', 'y', 'w', 'h']].values[0] == -1:
                    

            
            boxes = self.data_format_min_max(records[['x', 'y', 'w', 'h']].values, width, height)
            # print(f'{index} minmax {time.time() - t}')
            # t = time.time()
            
            if  self.smart_crop:
                image_lab = self.data_format_min_max_one_lab(image_lab)
                image, boxes = self.random_crop_with_bbox(image, image_lab, boxes, self.new_shape, teshold=0.5, format = self.data_format)
            
            if self.gaus:
                gaus_mask = self.gaus_mask(image, boxes)
                sample = {'img': image, 'mask': gaus_mask, 'annot':boxes} 
            elif self.class_mask:
                class_mask = self.create_class_mask(image, boxes)
                sample = {'img': image, 'mask': class_mask, 'annot':boxes} 
                # print(f'{index} mask {time.time() - t}')
                # t = time.time()
            else:
                sample = {'img': image, 'annot':boxes}
            
            if self.transforms:
                sample = self.transforms(sample)

            # print(f'{index} aug {time.time() - t}')
            # t = time.time()
            
            return sample
        
        elif self.mode == "test":
            
            # We just need to apply transoforms and return image
            if self.transforms:
                
                sample = {'img' : image}
                sample = self.transforms(sample)
                
            return sample    

    def __len__(self) -> int:
        return self.image_ids.shape[0]