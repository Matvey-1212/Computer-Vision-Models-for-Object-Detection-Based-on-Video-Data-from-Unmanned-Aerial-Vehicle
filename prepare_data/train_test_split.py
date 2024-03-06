import os
import random
import shutil

parent_folder = '/home/maantonov_1/VKR/data/crope_data/small/small_crop_train'
images_folder = os.path.join(parent_folder, 'images')
labels_folder = os.path.join(parent_folder, 'labels')

dest_folder_train = '/home/maantonov_1/VKR/data/for_yolo/small_crop_1024/train'
dest_folder_test = '/home/maantonov_1/VKR/data/for_yolo/small_crop_1024/val'
train_ratio = 0.8


image_files = [f for f in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, f))]


random.shuffle(image_files)
split_idx = int(len(image_files) * train_ratio)
train_files = image_files[:split_idx]
test_files = image_files[split_idx:]

if not os.path.exists(dest_folder_train):
    os.makedirs(dest_folder_train)

def move(names, source_path, destination_path):
    source_image_path = os.path.join(source_path, 'images') 
    source_label_path = os.path.join(source_path, 'labels') 
    
    destination_image_path = os.path.join(destination_path, 'images') 
    destination_label_path = os.path.join(destination_path, 'labels') 
    for name in names:
        name = os.path.splitext(name)[0]
        # if not os.path.isfile(os.path.join(image_path, name + '.jpg')) or os.path.isfile(os.path.join(label_path, name + '.txt')):
        #     print(name)
        #     continue
        source_file_img = os.path.join(source_image_path, name + '.JPG')
        destination_file_img = os.path.join(destination_image_path, name + '.JPG')
        
        source_file_lab = os.path.join(source_label_path, name + '.txt')
        destination_file_lab = os.path.join(destination_label_path, name + '.txt')
        
        if os.path.exists(source_file_img) and os.path.exists(source_file_lab):
            shutil.copy(source_file_img, destination_file_img)
            shutil.copy(source_file_lab, destination_file_lab)
            # print(destination_file_img)
            # print(destination_file_lab)
        else:
            print(f"Файл не найден: {source_file_img} or {source_file_lab}")

move(train_files, parent_folder, dest_folder_train)
move(test_files, parent_folder, dest_folder_test)

print("Файлы успешно перенесены.")