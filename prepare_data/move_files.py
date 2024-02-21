import os
import shutil

# Задаем путь к основной директории и папке назначения
base_dir = '/home/maantonov_1/VKR/data/main_data/train/labels'
destination_dir = '/home/maantonov_1/VKR/data/main_data/train2/labels'

# Создаем папку назначения, если она еще не существует
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)
count = 0
# Проходимся по всем поддиректориям в основной директории
for subdir, dirs, files in os.walk(base_dir):
    for file in files:
        # Проверяем, имеет ли файл расширение .jpg
        if file.endswith('.txt'):
            # Формируем полный путь к файлу
            file_path = os.path.join(subdir, file)
            # Перемещаем файл в папку назначения
            shutil.move(file_path, destination_dir)
            count+=1
            print(count)

print("Все файлы .txt были успешно перенесены в папку 'files'.")
print(count)
