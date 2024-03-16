from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
import os
import time
import cv2
import numpy as np
import torch

def flip_bboxes(bb, h_flip, v_flip, shapes, img_size = (1024, 1024)):
    for i in range(bb.shape[0]):
        bb[i, :shapes[i][0], 0] = (img_size[0] - bb[i, :shapes[i][0], 0]) * h_flip[i] + bb[i, :shapes[i][0], 0] * (1 - h_flip[i])
        bb[i, :shapes[i][0], 1] = (img_size[1] - bb[i, :shapes[i][0], 1]) * v_flip[i] + bb[i, :shapes[i][0], 1] * (1 - v_flip[i])
        bb[i, :shapes[i][0], 2] = (img_size[0] - bb[i, :shapes[i][0], 2]) * h_flip[i] + bb[i, :shapes[i][0], 2] * (1 - h_flip[i])
        bb[i, :shapes[i][0], 3] = (img_size[1] - bb[i, :shapes[i][0], 3]) * v_flip[i] + bb[i, :shapes[i][0], 3] * (1 - v_flip[i])
    
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

def visualize_and_save_images_cv2(images_tensor, bboxes_tensor, save_dir):
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

@pipeline_def( device_id=0)
def get_dali_pipeline_aug(images_dir, annotations_file):
    inputs, bboxes, labels, img_id = fn.readers.coco(
        file_root=images_dir,
        annotations_file=annotations_file,
        ltrb=True,
        random_shuffle=True,
        image_ids = True,
        name="Reader",
    )
    

    
    images = fn.decoders.image(inputs, device="mixed", output_type=types.RGB)

    
    apply_1 = fn.random.coin_flip(probability=0.2)
    apply_2 = fn.random.coin_flip(probability=0.2)
    apply_3 = fn.random.coin_flip(probability=0.2)
    apply_4 = fn.random.coin_flip(probability=0.2)
    apply_5 = fn.random.coin_flip(probability=0.5)
    apply_6 = fn.random.coin_flip(probability=0.5)
    
    images = fn.color_twist(images, brightness=fn.random.uniform(range=(0.8, 1.2)) * apply_1 + 1.0 * (1 - apply_1),
                            contrast=fn.random.uniform(range=(0.8, 1.2)) * apply_2 + 1.0 * (1 - apply_2), 
                            saturation=fn.random.uniform(range=(0.8, 1.2)) * apply_3 + 1.0 * (1 - apply_3),
                            hue=fn.random.uniform(range=(-0.2, 0.2)) * apply_4, device='gpu')
    
    # images = fn.color_twist(images, brightness=fn.random.uniform(range=(0.7, 1.3)),
    #                         contrast=fn.random.uniform(range=(0.7, 1.3)) , 
    #                         saturation=fn.random.uniform(range=(0.7, 1.3)),
    #                         hue=fn.random.uniform(range=(-0.1, 0.3)), device='gpu')
    
    images = fn.gaussian_blur(images, sigma=fn.random.uniform(range=(0.3, 1.7)), device='gpu')
    
    images = fn.noise.gaussian(images, stddev=fn.random.uniform(range=(1.0, 10.0)), device='gpu')


    horizontal_flip = fn.random.coin_flip(probability=0.5)  
    vertical_flip = fn.random.coin_flip(probability=0.5)  
    images = fn.flip(images, horizontal=horizontal_flip, vertical=vertical_flip, device='gpu')
    # bboxes = fn.bb_flip(bboxes, horizontal=horizontal_flip, vertical=vertical_flip, ltrb=True)
    
    bbox_shapes = fn.shapes(bboxes)
    bboxes = fn.pad(bboxes, fill_value=-1, axes=(0,1), shape=(60,4))

    images = fn.crop_mirror_normalize(images,
                                      mean=[0, 0, 0],
                                      std=[255, 255, 255], device='gpu')
    return images, bboxes, bbox_shapes, img_id, horizontal_flip, vertical_flip


@pipeline_def( device_id=0)
def get_dali_pipeline(images_dir, annotations_file):
    inputs, bboxes, labels, img_id = fn.readers.coco(
        file_root=images_dir,
        annotations_file=annotations_file,
        ltrb=True,
        random_shuffle=True,
        image_ids = True,
        name="Reader",
    )
    
    images = fn.decoders.image(inputs, device="mixed", output_type=types.RGB)
    
    bbox_shapes = fn.shapes(bboxes)
    bboxes = fn.pad(bboxes, fill_value=-1, axes=(0,1), shape=(60,4))
    
    # self
    # self.mean = np.array([[[0.501, 0.508, 0.497]]])
    # self.std = np.array([[[0.168, 0.174, 0.183]]])
    # imagenet
    # self.mean = np.array([[[0.485, 0.456, 0.406]]])
    # self.std = np.array([[[0.229, 0.224, 0.225]]])

    images = fn.crop_mirror_normalize(images,
                                      mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                      std=[0.229 * 255, 0.224 * 255, 0.225 * 255],)
    
    return images, bboxes, bbox_shapes, img_id

    
# main_dir = '/home/maantonov_1/VKR/data/crope_data/small/small_crop_val/'
# images_dir = main_dir + 'images'
# annotations_file = main_dir + 'annotations/annot.json'


# dali_iterator = DALIGenericIterator(
#     pipelines=[get_dali_pipeline_aug(images_dir = images_dir, annotations_file = annotations_file, batch_size = 16, num_threads = 4)],
#     output_map=['data', 'bboxe', 'bbox_shapes', 'img_id', 'horizontal_flip','vertical_flip'],
#     reader_name='Reader',
#     last_batch_policy=LastBatchPolicy.PARTIAL,
#     auto_reset=True,
#     dynamic_shape=True
# )

# t1 = time.time()
# for i, data in enumerate(dali_iterator):
    
#     x = data[0]['data']
#     bbox = data[0]['bboxe'].int()
#     z = data[0]['img_id']
#     h_flip = data[0]['horizontal_flip']
#     v_flip = data[0]['vertical_flip']
#     bb_shape = data[0]['bbox_shapes']
#     print(time.time()-t1)
#     t1 = time.time()
    

#     # bbox = flip_bboxes2(bbox, h_flip, v_flip, bb_shape)
    
    
    
#     # save_dir = '/home/maantonov_1/VKR/actual_scripts/retinanet_sep/utils/temp' 
#     # visualize_and_save_images_cv2(x, bbox, save_dir)
#     # break
    
#     # print(x.shape)
#     # print(h_flip.shape)
#     # print(v_flip.shape)
#     # print(bb_shape.shape)
#     # print(bbox)
#     # break

    
    

    





        




    

