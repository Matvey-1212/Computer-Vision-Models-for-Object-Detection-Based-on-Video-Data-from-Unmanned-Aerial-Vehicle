import os
import random
import cv2
import numpy as np
import torch
import torchvision
from scipy.interpolate import CubicSpline
import torch.nn.functional as F
import albumentations as A

class CopyPasteAugmentation:
    def __init__(self, image_dir_list,
                max_objects = 5,
                random_state = None,
                feather_amount = 111,
                num_points = 500,
                bb_pad = 0.0,
                transform = None):
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.max_objects = max_objects
        self.feather_amount = feather_amount
        self.num_points = num_points
        self.transform = transform
        self.bb_pad = bb_pad

        
        self.random_state = random_state
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
        
        self.image_dir_list = image_dir_list
        self.small_images = []
        self.annotation = []
        for dataset_path in self.image_dir_list:
            for path in os.listdir(os.path.join(dataset_path, 'images')):
                if path.endswith(('.png', '.jpg', '.JPG')):
                    try:
                        self.small_images.append(os.path.join(dataset_path, 'images', path))
                        self.annotation.append(os.path.join(dataset_path, 'labels', os.path.splitext(path)[0] + '.txt'))
                    except:
                        print('Error in spliting path to annot')
                        
                        
    def cyclic_moving_average(self, data, window_size):
        if window_size % 2 == 0:
            raise ValueError("Размер окна должен быть нечетным")

        extended_data = np.concatenate((data[-window_size//2:], data, data[:window_size//2]))
        weights = np.ones(window_size) / window_size
        smoothed_data = np.convolve(extended_data, weights, mode='valid')

        return smoothed_data

    def closed_distorted_circle(self, radius, num_points=100):

        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        base_points = np.linspace(0, 10, num_points // 10)
        random_modifiers = np.random.rand(num_points // 10) * 0.4 * radius
        random_modifiers[-1] = random_modifiers[0]

        cs = CubicSpline(base_points, random_modifiers)
        smooth_modifiers = cs(np.linspace(0, 10, num_points))

        modulated_radius = radius + smooth_modifiers
        x = modulated_radius * np.cos(angles)
        y = modulated_radius * np.sin(angles) 
        x = x * np.random.uniform(0.5, 1.5, 1)#x.shape[0])
        y = y * np.random.uniform(0.5, 1.5, 1)#y.shape[0])

        window_size = 9
        x = self.cyclic_moving_average(x, window_size)
        y = self.cyclic_moving_average(y, window_size)

        return x, y

    def create_feathered_mask_from_curve(self,  x, y, height, width, feather_amount=15):
        mask = np.zeros((height, width), dtype=np.uint8)
        curve_points = np.array([x, y]).T.reshape(-1, 1, 2).astype(np.int32)
        cv2.fillPoly(mask, [curve_points], 255)

        blur_mask = torch.tensor(cv2.GaussianBlur(mask, (feather_amount, feather_amount), 0))
        return blur_mask
    
    def create_feathered_mask(self, width, height, feather_edge=0.1):
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        x, y = np.meshgrid(x, y)

        distance = np.sqrt(x**2 + y**2)
        distance = np.clip(distance, 0, 1)

        alpha_mask = 1 - np.clip((distance - (1 - feather_edge)) / feather_edge, 0, 1)
        alpha_mask = (alpha_mask * 255).astype(np.uint8)  

        return torch.tensor(alpha_mask)

    def alpha_composite(self, foreground_tensor, background_tensor, alpha_channel, position=(0, 0)):

        fg_h, fg_w = foreground_tensor.shape[1], foreground_tensor.shape[2]
        x, y = position

        alpha_expanded = alpha_channel.expand_as(foreground_tensor)
#         print(x,y,fg_h, fg_w)
        bg_slice = background_tensor[:, y:y+fg_h, x:x+fg_w]
        
#         print(alpha_expanded.shape)
        

        composite = alpha_expanded * foreground_tensor + (1 - alpha_expanded) * bg_slice

        result_tensor = background_tensor 
        result_tensor[:, y:y+fg_h, x:x+fg_w] = composite

        return result_tensor

    def apply_augmentation(self, background_images, gt_annots):
        
        batch_size = background_images.shape[0]
        bg_shape = background_images.shape[2:]
        
        for i in range(batch_size):
            num_objects = random.randint(1, self.max_objects)  
            
            gt_ereas = []
            new_centers_list = []
            
            if (gt_annots[i, :, -1] != -1).sum() != 0:
                # print('!')
                continue
            
            
            small_image_list = []
            for k in range(num_objects):
                
                # for val in gt_annots[i]:
                #     if val[-1] == -1:
                #         continue
                
                
                index = random.randint(0, len(self.small_images) - 1)
                local_image_path = self.small_images[index]
                local_annot_path = self.annotation[index]
                

                local_image = torchvision.io.read_image(local_image_path)
                
                with open(local_annot_path) as f:
                    local_annot = f.readlines()
                for d, val in enumerate(local_annot):
                    local_annot[d] = val.split(' ')
                    if '\n' in local_annot[d][-1]:
                        local_annot[d][-1] = local_annot[d][-1][:-1]
                
                l_height, l_width = local_image.shape[1:]
                
                random_scale = np.random.uniform(0.2, 1.2, 1)
                
                box = local_annot[0]
                _,_,_, w, h = box

                w, h = float(w) * l_width, float(h) * l_height
                
                
                new_scaled_w = min(40, max(w * random_scale, 7))/w
                new_scaled_h = min(60, max(h * random_scale, 15))/h
                
                local_image = F.interpolate(local_image.unsqueeze(0),  
                                          size=(int(l_height * new_scaled_h), int(l_width * new_scaled_w)), 
                                          mode='bilinear',  
                                          align_corners=False).squeeze(0).to(self.device).float()/255

                l_height, l_width = local_image.shape[1:]
                
                new_x = np.random.randint(0, bg_shape[1]-l_width, 1)[0]
                new_y = np.random.randint(0, bg_shape[0]-l_height, 1)[0]
                
                flag_0 = False
                for d, val in enumerate(gt_annots[i]):
                    if val[-1] != -1:
                        x_t1 = gt_annots[i][d][0]
                        y_t1 = gt_annots[i][d][1]
                        x_t2 = gt_annots[i][d][2]
                        y_t2 = gt_annots[i][d][3]
                        
                        x_c = int((x_t2 + x_t1)/2)
                        y_c = int((y_t2 + y_t1)/2)
                        
                        if new_x - 10 <= x_c <= new_x + l_width + 10:
                            flag_0 = True
                        if new_y - 10 <= y_c <= new_y + l_height + 10:
                            flag_0 = True
                if flag_0:
                    continue
                        
                box = local_annot[0]
                
                class_id, x_center, y_center, w, h = box
                class_id, x_center, y_center, w, h = float(class_id), float(x_center), float(y_center), float(w), float(h)

                x_center, y_center, w, h = x_center * l_width, y_center * l_height, w * l_width, h * l_height

                x1 = int(x_center - w / 2)
                y1 = int(y_center - h / 2)

                x2 = int(x_center + w / 2)
                y2 = int(y_center + h / 2)
                
                if np.random.rand() < 0.5:
                    local_image = local_image.flip(dims=[-1])#[:, ::-1, :]
                if np.random.rand() < 0.5:
                    local_image = local_image.flip(dims=[-2])#[:, :, ::-1]

                # radius = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) * 0.5
                
                # x_closed_distorted, y_closed_distorted = self.closed_distorted_circle(radius, num_points= self.num_points)
                # mask = self.create_feathered_mask_from_curve(x_closed_distorted + l_width // 2, y_closed_distorted + l_height // 2, l_height, l_width, feather_amount=self.feather_amount).to(self.device)
#                 mask = self.create_feathered_mask(l_width, l_height, 0.5).to(self.device)
#                 mask = torch.ones((l_height, l_width)) 
    
                # background_images[i] = self.alpha_composite(local_image, background_images[i], mask/255, position=(new_x, new_y))
        
#                 if self.transform is not None:
#                     background_images[i] = torch.tensor(self.transform(image=background_images[i].permute(1,2,0).numpy())['image']).permute(2,0,1)

                background_images[i,:,new_y:l_height+new_y,new_x:l_width+new_x] = local_image
    
                for d, val in enumerate(gt_annots[i]):
                    if val[-1] == -1:
                        gt_annots[i][d][0] = new_x + x1
                        gt_annots[i][d][1] = new_y + y1
                        gt_annots[i][d][2] = new_x + x2
                        gt_annots[i][d][3] = new_y + y2
                        gt_annots[i][d][4] = 0
                        
                        
                        if self.bb_pad > 0:
                            gt_annots[i][d][0] = max(0, min(gt_annots[i][d][0] - self.bb_pad * w, bg_shape[1]))
                            gt_annots[i][d][1] = max(0, min(gt_annots[i][d][1] - self.bb_pad * h, bg_shape[0]))
                            gt_annots[i][d][2] = max(0, min(gt_annots[i][d][2] + self.bb_pad * w, bg_shape[1]))
                            gt_annots[i][d][3] = max(0, min(gt_annots[i][d][3] + self.bb_pad * h, bg_shape[0]))
                            
                        break

        return background_images, gt_annots
            

