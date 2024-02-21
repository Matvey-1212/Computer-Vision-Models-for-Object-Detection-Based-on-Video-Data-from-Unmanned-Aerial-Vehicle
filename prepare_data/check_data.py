import os

path = '/home/maantonov_1/VKR/data/small_train/test/labels'

image_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
print(len(image_files))

path = '/home/maantonov_1/VKR/data/small_train/train/labels'

image_files1 = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
print(len(image_files1))

print(len(image_files)+len(image_files1))