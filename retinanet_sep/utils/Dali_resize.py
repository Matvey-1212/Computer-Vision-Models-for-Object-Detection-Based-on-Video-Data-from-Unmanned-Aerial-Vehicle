from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
import os
import time
import cv2
import numpy as np
import torch
import albumentations as A

transform_90 = A.Compose([
    A.Rotate(limit=[90, 90], p=1.0)
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

transform_270 = A.Compose([
    A.Rotate(limit=[-90, -90], p=1.0)
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

mean=[0.485 * 255, 0.456 * 255, 0.406 * 255]
std=[0.229 * 255, 0.224 * 255, 0.225 * 255]

# mean=[0.4959 * 255, 0.5021 * 255, 0.4903 * 255]
# std=[0.1756 * 255, 0.218034 * 255, 0.1880 * 255]

# mean = [0.4856 * 255, 0.4858 * 255, 0.4647 * 255]
# std = [0.1671 * 255, 0.1731 * 255, 0.1825 * 255]

def flip_bboxes(bb, h_flip, v_flip, shapes, img_size = (1024, 1024)):
    
    h, w = img_size
    
    for i in range(bb.shape[0]):
        b0 = torch.clone(bb[i, :shapes[i][0], 0])
        b1 = torch.clone(bb[i, :shapes[i][0], 1])
        b2 = torch.clone(bb[i, :shapes[i][0], 2])
        b3 = torch.clone(bb[i, :shapes[i][0], 3])
        
        
        bb[i, :shapes[i][0], 0] = (img_size[1] - b2) * h_flip[i] + bb[i, :shapes[i][0], 0] * (1 - h_flip[i]).clip(0, w-1)
        bb[i, :shapes[i][0], 1] = (img_size[0] - b3) * v_flip[i] + bb[i, :shapes[i][0], 1] * (1 - v_flip[i]).clip(0, h-1)
        bb[i, :shapes[i][0], 2] = (img_size[1] - b0) * h_flip[i] + bb[i, :shapes[i][0], 2] * (1 - h_flip[i]).clip(0, w-1)
        bb[i, :shapes[i][0], 3] = (img_size[0] - b1) * v_flip[i] + bb[i, :shapes[i][0], 3] * (1 - v_flip[i]).clip(0, h-1)
        # print(img_size)
        # print(b0)
        # print(b1)
        # print(b2)
        # print(b3)
        # print(bb[i, :shapes[i][0], 0])
        # print(bb[i, :shapes[i][0], 1])
        # print(bb[i, :shapes[i][0], 2])
        # print(bb[i, :shapes[i][0], 3])
        # print()
    
    return bb

def rotate_bboxes(bb, rotate, shapes, img_size = (1024, 1024)):
    
    h, w = img_size
    for i in range(bb.shape[0]):
        b0 = torch.clone(bb[i, :shapes[i][0], 0]).clip(0, w-1)
        b1 = torch.clone(bb[i, :shapes[i][0], 1]).clip(0, h-1)
        b2 = torch.clone(bb[i, :shapes[i][0], 2]).clip(0, w-1)
        b3 = torch.clone(bb[i, :shapes[i][0], 3]).clip(0, h-1)
        
        if rotate[i] == 0:
            continue
        
        if rotate[i] == 90:
            new_x_min = b1
            new_y_min = w - b2 - 1
            new_x_max = b3
            new_y_max = w - b0 - 1  
        elif rotate[i] == -90:
            new_x_min = h - b3 - 1
            new_y_min = b0
            new_x_max = h - b1 - 1
            new_y_max = b2
            
        bb[i, :shapes[i][0], 0] = torch.clip(new_x_min, 0, w-1)
        bb[i, :shapes[i][0], 1] = torch.clip(new_y_min, 0, h-1)
        bb[i, :shapes[i][0], 2] = torch.clip(new_x_max, 0, w-1)
        bb[i, :shapes[i][0], 3] = torch.clip(new_y_max, 0, h-1)
    
    return bb

def rotate_bboxes2(bb, rotate, shapes, img_size = (1024, 1024)):
    
    h, w = img_size
    for i in range(bb.shape[0]):
        if float(shapes[i][0]) == 0:
            continue
        bb_t = torch.clone(bb[i, :shapes[i][0]])
        
        if rotate[i] == 0:
            continue
        
        if rotate[i] == 90:
            transformed = transform_90(image=np.zeros((h,w)),bboxes=bb_t, class_labels=np.zeros(shapes[i][0]))
            transformed_bboxes = transformed['bboxes']
            
        elif rotate[i] == -90:
            transformed = transform_270(image=np.zeros((h,w)),bboxes=bb_t, class_labels=np.zeros(shapes[i][0]))
            transformed_bboxes = transformed['bboxes']
            
        # print(torch.tensor(transformed_bboxes).shape)
        # print(shapes[i][0])
        bb[i, :shapes[i][0]] = torch.tensor(transformed_bboxes)
    
    return bb

def flip_bboxes2(bb, h_flip, v_flip, bb_shape, img_size=(1024, 1024)):
    h_flip_expanded = h_flip.unsqueeze(1).unsqueeze(-1)
    v_flip_expanded = v_flip.unsqueeze(1).unsqueeze(-1)

    indices = torch.arange(bb.size(1), device=bb.device).unsqueeze(0).expand(bb.size(0), -1).unsqueeze(-1)
    active_mask = (indices < bb_shape[:, 0].unsqueeze(1).unsqueeze(-1)).squeeze()
    

    bb[:, :, 0] = torch.where(active_mask, img_size[0] * h_flip_expanded[:, :, 0] - bb[:, :, 0] * (2 * h_flip_expanded[:, :, 0] - 1), bb[:, :, 0])
    bb[:, :, 1] = torch.where(active_mask, img_size[1] * v_flip_expanded[:, :, 0] - bb[:, :, 1] * (2 * v_flip_expanded[:, :, 0] - 1), bb[:, :, 1])
    bb[:, :, 2] = torch.where(active_mask, img_size[0] * h_flip_expanded[:, :, 0] - bb[:, :, 2] * (2 * h_flip_expanded[:, :, 0] - 1), bb[:, :, 2])
    bb[:, :, 3] = torch.where(active_mask, img_size[1] * v_flip_expanded[:, :, 0] - bb[:, :, 3] * (2 * v_flip_expanded[:, :, 0] - 1), bb[:, :, 3])
    

    return bb

def visualize_and_save_images_cv2(images_tensor, bboxes_tensor, sizes, save_dir):
    # Убедитесь, что директория для сохранения существует
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Конвертировать тензоры в numpy массивы
    images_np = images_tensor.cpu().numpy()
    bboxes_np = bboxes_tensor.cpu().numpy()

    for i, image in enumerate(images_np):
        # Преобразование из формата [C, H, W] в [H, W, C] и нормализация [0, 1] -> [0, 255]
        image = np.transpose(image, (1, 2, 0))  # из [C, H, W] в [H, W, C]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # из RGB в BGR для OpenCV
        image = (image).astype(np.uint8)  # Нормализация из [0, 1] в [0, 255]
        
        # w_c = 1024 / sizes[i][1]
        # h_c = 1024 / sizes[i][0]

        # Рисование ограничивающих рамок
        for bbox in bboxes_np[i]:
            # Проверка на паддинг
            if (bbox == -1).all():
                continue
            # if h_flip[i] == 1:
            #     bbox[0]

            start_point = (int(bbox[0]), int(bbox[1]))  # (x_min, y_min)
            end_point = (int(bbox[2]), int(bbox[3]))  # (x_max, y_max)
            color = (0, 255, 0)  # Зеленый цвет в BGR
            thickness = 2  # Толщина линии рамки
            image = cv2.rectangle(image, start_point, end_point, color, thickness)

        # Сохранение изображения
        cv2.imwrite(os.path.join(save_dir, f'image_{i}_{z[i][0]}.jpg'), image)
        print(f'image_{i}_{z[i][0]}.jpg')
        
def resize_bb(bboxes_tensor, sizes, bb_pad = 0.0 ,new_shape = (1024, 1024)):
    for i in range(bboxes_tensor.shape[0]):
        
        width  = new_shape[1]
        height = new_shape[0]
        
        w_c = width / sizes[i][1]
        h_c = height / sizes[i][0]

        for j in range(bboxes_tensor[i].shape[0]):
            if (bboxes_tensor[i][j] == -1).all():
                continue
            
            dx = bboxes_tensor[i][j][2] - bboxes_tensor[i][j][0]
            dy = bboxes_tensor[i][j][3] - bboxes_tensor[i][j][1]
            
            bboxes_tensor[i][j][0] = max(0, min(int(bboxes_tensor[i][j][0] * w_c - bb_pad * dx * w_c), width))
            bboxes_tensor[i][j][1] = max(0, min(int(bboxes_tensor[i][j][1] * h_c - bb_pad * dy * h_c), height))
            bboxes_tensor[i][j][2] = max(0, min(int(bboxes_tensor[i][j][2] * w_c + bb_pad * dx * w_c), width))
            bboxes_tensor[i][j][3] = max(0, min(int(bboxes_tensor[i][j][3] * h_c + bb_pad * dy * h_c), height))
    return bboxes_tensor

def up_bb(bboxes_tensor, sizes, bb_pad = 0.0 , shape = (1024, 1024)):
    for i in range(bboxes_tensor.shape[0]):
        
        width  = shape[1]
        height = shape[0]

        for j in range(bboxes_tensor[i].shape[0]):
            if (bboxes_tensor[i][j] == -1).all():
                continue
            
            dx = bboxes_tensor[i][j][2] - bboxes_tensor[i][j][0]
            dy = bboxes_tensor[i][j][3] - bboxes_tensor[i][j][1]
            
            bboxes_tensor[i][j][0] = max(0, min(int(bboxes_tensor[i][j][0]  - bb_pad * dx), width))
            bboxes_tensor[i][j][1] = max(0, min(int(bboxes_tensor[i][j][1]  - bb_pad * dy), height))
            bboxes_tensor[i][j][2] = max(0, min(int(bboxes_tensor[i][j][2]  + bb_pad * dx), width))
            bboxes_tensor[i][j][3] = max(0, min(int(bboxes_tensor[i][j][3]  + bb_pad * dy), height))
    return bboxes_tensor

@pipeline_def( device_id=0)
def get_dali_pipeline_aug(images_dir, annotations_file, resize_dims=(1024, 1024), seed = 42):
    inputs, bboxes, labels, img_id = fn.readers.coco(
        file_root=images_dir,
        annotations_file=annotations_file,
        ltrb=True,
        random_shuffle=True,
        image_ids = True,
        name="Reader",
        seed = seed
    )
    

    original_sizes = fn.peek_image_shape(inputs)
    images = fn.decoders.image(inputs, device="mixed", output_type=types.RGB)

    images = fn.resize(images, resize_x=resize_dims[0], resize_y=resize_dims[1], device='gpu')

    

    
    images = fn.color_twist(images, brightness=fn.random.uniform(range=(0.5, 1.5), seed=seed+1),
                            contrast=fn.random.uniform(range=(0.5, 1.5), seed=seed+2) , 
                            saturation=fn.random.uniform(range=(0.5, 1.5), seed=seed+3),
                            hue=fn.random.uniform(range=(-0.1, 0.3), seed=seed+4), device='gpu')
    
    images = fn.gaussian_blur(images, sigma=fn.random.uniform(range=(0.3, 1.7), seed=seed+5), device='gpu')
    
    images = fn.noise.gaussian(images, stddev=fn.random.uniform(range=(1.0, 10.0), seed=seed+6), device='gpu')


    horizontal_flip = fn.random.coin_flip(probability=0.5, seed=seed+7)  
    vertical_flip = fn.random.coin_flip(probability=0.5, seed=seed+8)  
    images = fn.flip(images, horizontal=horizontal_flip, vertical=vertical_flip, device='gpu')
    # bboxes = fn.bb_flip(bboxes, horizontal=horizontal_flip, vertical=vertical_flip, ltrb=True)
    
    # angles = fn.random.uniform(values=[0, 90, -90], seed=seed+9)  # Выбор из 0, 90, -90 градусов
    # images = fn.rotate(images, angle=angles, keep_size=True)
    
    bbox_shapes = fn.shapes(bboxes)
    bboxes = fn.pad(bboxes, fill_value=-1, axes=(0,1), shape=(60,4))

    images = fn.crop_mirror_normalize(images, 
                                      mean=mean,
                                      std=std, device='gpu')
    return images, bboxes, bbox_shapes, img_id, horizontal_flip, vertical_flip, original_sizes#, angles

@pipeline_def( device_id=0)
def get_dali_pipeline_aug_rotate(images_dir, annotations_file, resize_dims=(1024, 1024), seed = 42):
    inputs, bboxes, labels, img_id = fn.readers.coco(
        file_root=images_dir,
        annotations_file=annotations_file,
        ltrb=True,
        random_shuffle=True,
        image_ids = True,
        name="Reader",
        seed = seed
    )
    

    original_sizes = fn.peek_image_shape(inputs)
    images = fn.decoders.image(inputs, device="mixed", output_type=types.RGB)

    images = fn.resize(images, resize_x=resize_dims[0], resize_y=resize_dims[1], device='gpu')

    

    
    images = fn.color_twist(images, brightness=fn.random.uniform(range=(0.5, 1.5)),
                            contrast=fn.random.uniform(range=(0.5, 1.5)) , 
                            saturation=fn.random.uniform(range=(0.5, 1.5)),
                            hue=fn.random.uniform(range=(-0.1, 0.3)), device='gpu')
    
    images = fn.gaussian_blur(images, sigma=fn.random.uniform(range=(0.3, 1.7)), device='gpu')
    
    images = fn.noise.gaussian(images, stddev=fn.random.uniform(range=(1.0, 10.0)), device='gpu')


    horizontal_flip = fn.random.coin_flip(probability=0.5)  
    vertical_flip = fn.random.coin_flip(probability=0.5)  
    images = fn.flip(images, horizontal=horizontal_flip, vertical=vertical_flip, device='gpu')
    # bboxes = fn.bb_flip(bboxes, horizontal=horizontal_flip, vertical=vertical_flip, ltrb=True)
    
    angles = fn.random.uniform(values=[0, 90, -90])  # Выбор из 0, 90, -90 градусов
    images = fn.rotate(images, angle=angles, keep_size=True)
    
    bbox_shapes = fn.shapes(bboxes)
    bboxes = fn.pad(bboxes, fill_value=-1, axes=(0,1), shape=(60,4))

    images = fn.crop_mirror_normalize(images, 
                                      mean=mean,
                                      std=std, device='gpu')
    return images, bboxes, bbox_shapes, img_id, horizontal_flip, vertical_flip, original_sizes, angles


@pipeline_def( device_id=0)
def get_dali_pipeline_small_aug(images_dir, annotations_file, resize_dims=(1024, 1024), seed = 42):
    inputs, bboxes, labels, img_id = fn.readers.coco(
        file_root=images_dir,
        annotations_file=annotations_file,
        ltrb=True,
        random_shuffle=True,
        image_ids = True,
        name="Reader",
        seed = seed
    )
    

    original_sizes = fn.peek_image_shape(inputs)
    images = fn.decoders.image(inputs, device="mixed", output_type=types.RGB)

    images = fn.resize(images, resize_x=resize_dims[0], resize_y=resize_dims[1], device='gpu')


    horizontal_flip = fn.random.coin_flip(probability=0.5, seed=seed+7)  
    vertical_flip = fn.random.coin_flip(probability=0.5, seed=seed+8)  
    images = fn.flip(images, horizontal=horizontal_flip, vertical=vertical_flip, device='gpu')
    # bboxes = fn.bb_flip(bboxes, horizontal=horizontal_flip, vertical=vertical_flip, ltrb=True)
    
    bbox_shapes = fn.shapes(bboxes)
    bboxes = fn.pad(bboxes, fill_value=-1, axes=(0,1), shape=(60,4))

    images = fn.crop_mirror_normalize(images,device='gpu')
    return images, bboxes, bbox_shapes, img_id, horizontal_flip, vertical_flip, original_sizes


@pipeline_def( device_id=0)
def get_dali_pipeline(images_dir, annotations_file, resize_dims=(1024, 1024), seed = 42):
    inputs, bboxes, labels, img_id = fn.readers.coco(
        file_root=images_dir,
        annotations_file=annotations_file,
        ltrb=True,
        random_shuffle=False,
        image_ids = True,
        name="Reader",
        seed = seed
    )
    original_sizes = fn.peek_image_shape(inputs)
    images = fn.decoders.image(inputs, device="mixed", output_type=types.RGB)
    
    images = fn.resize(images, resize_x=resize_dims[0], resize_y=resize_dims[1], device='gpu')
    
    bbox_shapes = fn.shapes(bboxes)
    bboxes = fn.pad(bboxes, fill_value=-1, axes=(0,1), shape=(60,4))
    


    images = fn.crop_mirror_normalize(images,
                                      mean=mean,
                                      std=std, device='gpu')
    
    return images, bboxes, bbox_shapes, img_id, original_sizes


@pipeline_def( device_id=0)
def get_dali_pipeline_two_stages(images_dir, annotations_file, resize_dims=(1024, 1024), seed = 42):
    inputs, bboxes, labels, img_id = fn.readers.coco(
        file_root=images_dir,
        annotations_file=annotations_file,
        ltrb=True,
        random_shuffle=False,
        image_ids = True,
        name="Reader",
        seed = seed
    )
    original_sizes = fn.peek_image_shape(inputs)
    images = fn.decoders.image(inputs, device="mixed", output_type=types.RGB)
    
    images_resized = fn.resize(images, resize_x=resize_dims[0], resize_y=resize_dims[1])
    
    # bbox_shapes = fn.shapes(bboxes)
    # bboxes = fn.pad(bboxes, fill_value=-1, axes=(0,1), shape=(60,4))
    



    images = fn.crop_mirror_normalize(images,)
                                    #   mean=mean,
                                    #   std=std, device='gpu')
    
    images_resized = fn.crop_mirror_normalize(images_resized,
                                      mean=mean,
                                      std=std, device='gpu')
    
    return images, images_resized, bboxes, img_id #bbox_shapes, img_id#, original_sizes

@pipeline_def( device_id=0)
def get_dali_pipeline_two_stages_faster(images_dir, annotations_file, resize_dims=(1024, 1024), seed = 42):
    inputs, bboxes, labels, img_id = fn.readers.coco(
        file_root=images_dir,
        annotations_file=annotations_file,
        ltrb=True,
        random_shuffle=False,
        image_ids = True,
        name="Reader",
        seed = seed
    )
    original_sizes = fn.peek_image_shape(inputs)
    images = fn.decoders.image(inputs, device="mixed", output_type=types.RGB)
    
    images_resized = fn.resize(images, resize_x=resize_dims[0], resize_y=resize_dims[1])
    
    # bbox_shapes = fn.shapes(bboxes)
    # bboxes = fn.pad(bboxes, fill_value=-1, axes=(0,1), shape=(60,4))
    



    images = fn.crop_mirror_normalize(images,)
                                    #   mean=mean,
                                    #   std=std, device='gpu')
    
    images_resized = fn.crop_mirror_normalize(images_resized,)
                                    #   mean=mean,
                                    #   std=std, device='gpu')
    
    return images, images_resized, bboxes, img_id #bbox_shapes, img_id#, original_sizes



@pipeline_def( device_id=0)
def get_dali_pipeline_aug_rotate_no_resize(images_dir, annotations_file, seed = 42):
    inputs, bboxes, labels, img_id = fn.readers.coco(
        file_root=images_dir,
        annotations_file=annotations_file,
        ltrb=True,
        random_shuffle=True,
        image_ids = True,
        name="Reader",
        seed = seed
    )
    

    images = fn.decoders.image(inputs, device="mixed", output_type=types.RGB)    

    
    images = fn.color_twist(images, brightness=fn.random.uniform(range=(0.5, 1.5), seed=seed+1),
                            contrast=fn.random.uniform(range=(0.5, 1.5), seed=seed+2) , 
                            saturation=fn.random.uniform(range=(0.5, 1.5), seed=seed+3),
                            hue=fn.random.uniform(range=(-0.1, 0.3), seed=seed+4), device='gpu')
    
    images = fn.gaussian_blur(images, sigma=fn.random.uniform(range=(0.3, 1.7), seed=seed+5), device='gpu')
    
    images = fn.noise.gaussian(images, stddev=fn.random.uniform(range=(1.0, 10.0), seed=seed+6), device='gpu')


    horizontal_flip = fn.random.coin_flip(probability=0.5, seed=seed+7)  
    vertical_flip = fn.random.coin_flip(probability=0.5, seed=seed+8)  
    images = fn.flip(images, horizontal=horizontal_flip, vertical=vertical_flip, device='gpu')
    # bboxes = fn.bb_flip(bboxes, horizontal=horizontal_flip, vertical=vertical_flip, ltrb=True)
    
    angles = fn.random.uniform(values=[0, 90, -90], seed=seed+9)  # Выбор из 0, 90, -90 градусов
    images = fn.rotate(images, angle=angles, keep_size=True)
    
    bbox_shapes = fn.shapes(bboxes)
    bboxes = fn.pad(bboxes, fill_value=-1, axes=(0,1), shape=(60,4))

    images = fn.crop_mirror_normalize(images, 
                                      mean=mean,
                                      std=std, device='gpu')
    return images, bboxes, bbox_shapes, img_id, horizontal_flip, vertical_flip, angles

@pipeline_def( device_id=0)
def get_dali_pipeline_no_resize(images_dir, annotations_file, seed = 42):
    inputs, bboxes, labels, img_id = fn.readers.coco(
        file_root=images_dir,
        annotations_file=annotations_file,
        ltrb=True,
        random_shuffle=False,
        image_ids = True,
        name="Reader",
        seed = seed
    )
    original_sizes = fn.peek_image_shape(inputs)
    images = fn.decoders.image(inputs, device="mixed", output_type=types.RGB)
    
    bbox_shapes = fn.shapes(bboxes)
    bboxes = fn.pad(bboxes, fill_value=-1, axes=(0,1), shape=(60,4))
    


    images = fn.crop_mirror_normalize(images,
                                      mean=mean,
                                      std=std, device='gpu')
    
    return images, bboxes, bbox_shapes, img_id
    
# # main_dir = '/home/maantonov_1/VKR/data/crope_data/small/small_crop_val/'
# # main_dir = '/home/maantonov_1/VKR/data/crope_data/main/crop_train_1024/'
# main_dir = '/home/maantonov_1/VKR/data/main_data/train/'

# images_dir = main_dir + 'images'
# annotations_file = main_dir + 'true_train_annot/annot.json'


# dali_iterator = DALIGenericIterator(
#     pipelines=[get_dali_pipeline_aug(images_dir = images_dir, annotations_file = annotations_file, batch_size = 16, num_threads = 4)],
#     output_map=['data', 'bboxe', 'bbox_shapes', 'img_id', 'horizontal_flip','vertical_flip', 'original_sizes', 'angles'],
#     reader_name='Reader',
#     last_batch_policy=LastBatchPolicy.PARTIAL,
#     auto_reset=True,
#     dynamic_shape=True
# )

# dali_iterator_val = DALIGenericIterator(
#     pipelines=[get_dali_pipeline(images_dir = images_dir, annotations_file = annotations_file, batch_size = 16, num_threads = 1)],
#     output_map=['data', 'bboxe', 'bbox_shapes', 'img_id', 'original_sizes'],
#     reader_name='Reader',
#     last_batch_policy=LastBatchPolicy.PARTIAL,
#     auto_reset=True,
#     dynamic_shape=True
# )

# tiem_list = []
# t1 = time.time()

# sums = torch.tensor([0.0, 0.0, 0.0]).cuda()
# squared_sums = torch.tensor([0.0, 0.0, 0.0]).cuda()
# n_pixels = 0

# print('!')
# for i, data in enumerate(dali_iterator):
    
#     x = data[0]['data'].float()
#     bbox = data[0]['bboxe'].int()
#     z = data[0]['img_id']
#     h_flip = data[0]['horizontal_flip']
#     v_flip = data[0]['vertical_flip']
#     bb_shape = data[0]['bbox_shapes']
#     original_sizes = data[0]['original_sizes']
#     angles = data[0]['angles']

#     print(angles)

#     bbox = resize_bb(bbox, original_sizes, bb_pad = 0.5, new_shape = (1024, 1024))
#     bbox = flip_bboxes(bbox, h_flip, v_flip, bb_shape)
#     bbox = rotate_bboxes(bbox, angles, bb_shape)
    
#     save_dir = '/home/maantonov_1/VKR/actual_scripts/retinanet_sep/utils/temp' 
#     visualize_and_save_images_cv2(x, bbox, original_sizes, save_dir)
#     print(data)
    
#     # sums += x.sum(dim=[0, 2, 3])

#     # squared_sums += (x ** 2).sum(dim=[0, 2, 3])

#     # n_pixels += x.size(0) * x.size(2) * x.size(3)
#     exit()


# means = sums / n_pixels


# stds = (squared_sums / n_pixels - means ** 2) ** 0.5
    
# print('Средние значения каналов:', means)
# print('Стандартные отклонения каналов:', stds)
    
# # #     bbox = flip_bboxes(bbox, h_flip, v_flip, bb_shape)
# # #     # tiem_list.append(time.time()-t1)
# # #     # print(time.time()-t1)
# # # #     t1 = time.time()
# # # # print(np.mean(tiem_list))
# # #     for i in range(bbox.shape[0]):
# # #         for j in bbox[i]:
# # #             if j[0] != -1:
# # #                 # print(j)
# # #                 if j[0] > j[2] or j[1] > j[3]:
# # #                     print(z[i])
# # #                     print(j)

# # #     # bbox = flip_bboxes2(bbox, h_flip, v_flip, bb_shape)
    
    
    
# # #     # save_dir = '/home/maantonov_1/VKR/actual_scripts/retinanet_sep/utils/temp' 
# # #     # visualize_and_save_images_cv2(x, bbox, save_dir)
# # #     # break
    
# # #     # print(x.shape)
# # #     # print(h_flip.shape)
# # #     # print(v_flip.shape)
# # #     # print(bb_shape.shape)
# # #     # print(bbox)
# # #     # break

