import pandas as pd


from utils.datasetLLAD import LLAD




DIR_TRAIN = '/home/maantonov_1/VKR/data/main_data/train/images'
train_df = pd.read_csv('/home/maantonov_1/VKR/data/main_data/train/train_main.csv')


train_dataset = LLAD(train_df, DIR_TRAIN, mode = "train", smart_crop = False, new_shape = (2048, 2048))


print(f'dataset Created', flush=True)

r = 0
g = 0
b = 0
r2 = 0
g2 = 0
b2 = 0
N = len(train_dataset)
for i in range(N):
    print(f'{i}/{N}')
    image = train_dataset[i]['img']
    h, w, c = image.shape

    r += image[:,:,0].sum() / (h*w)
    g += image[:,:,1].sum() / (h*w)
    b += image[:,:,2].sum() / (h*w)

    r2 += (image[:,:,0] ** 2).sum() / (h*w) 
    g2 += (image[:,:,1] ** 2).sum() / (h*w) 
    b2 += (image[:,:,2] ** 2).sum() / (h*w) 
    
r_mean = r / N
g_mean = g / N    
b_mean = b / N
    
s_r = ((r2 - N * r_mean ** 2 )/(N-1))**(0.5)
s_g = ((g2 - N * g_mean ** 2 )/(N-1))**(0.5)
s_b = ((b2 - N * b_mean ** 2 )/(N-1))**(0.5)

print(f'mean {r_mean}, {g_mean}, {b_mean}')
print(f's {s_r}, {s_g}, {s_b}')