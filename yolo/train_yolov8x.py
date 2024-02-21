import torch
from ultralytics import YOLO
from clearml import Task

Task.set_offline(offline_mode=True)
task = Task.init(project_name='VKR', task_name='yolov8x')


model = YOLO('/home/maantonov_1/VKR/weights/yolov8_visdrone/yolov8x.pt')
 
# Training.
results = model.train(
   data='/home/maantonov_1/VKR/actual_scripts/yolo/data_conf/data_conf_big.yml',
   imgsz=[8000.6000],
   epochs=200,
   batch=1,
   name='yolov8x',
   device='0',
   task= 'detect', # (str) YOLO task, i.e. detect, segment, classify, pose
   mode= 'train',
   lr0 = 0.001,
   lrf = 0.01,
   plots = False,
   format = '',
   single_cls= True,
   workers = 4,
   momentum= 0.937, # (float) SGD momentum/Adam beta1
   weight_decay= 0.0005, # (float) optimizer weight decay 5e-4
   warmup_epochs= 3.0, # (float) warmup epochs (fractions ok)
   warmup_momentum= 0.8, # (float) warmup initial momentum
   warmup_bias_lr= 0.1, # (float) warmup initial bias lr
   box =7.5, # (float) box loss gain
   label_smoothing= 0.0, # (float) label smoothing (fraction)
   hsv_h =0.015, # (float) image HSV-Hue augmentation (fraction)
   hsv_s =0.7, # (float) image HSV-Saturation augmentation (fraction)
   hsv_v =0.4, # (float) image HSV-Value augmentation (fraction)
   degrees =0.0, # (float) image rotation (+/- deg)
   translate= 0.1, # (float) image translation (+/- fraction)
   scale =0.1, # (float) image scale (+/- gain)
   shear =5, # (float) image shear (+/- deg)
   perspective= 0.001, # (float) image perspective (+/- fraction), range 0-0.001
   flipud =0.5, # (float) image flip up-down (probability)
   fliplr =0.5, # (float) image flip left-right (probability)
   mosaic= 0.1, # (float) image mosaic (probability)
   mixup= 0.1, # (float) image mixup (probability)
   auto_augment= 'randaugment')

# # Загрузка предварительно обученной модели
# model = YOLO('/home/maantonov_1/VKR/actual_scripts/yolo/configs/yolov8n.yml')

# # Проверка доступности GPU
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model.to(device)

# model.train()
