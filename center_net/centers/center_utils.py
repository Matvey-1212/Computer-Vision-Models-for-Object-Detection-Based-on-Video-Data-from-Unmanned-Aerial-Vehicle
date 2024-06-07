import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import f1_score


input_size = 1920
IN_SCALE = 1024//input_size 
MODEL_SCALE = 4
batch_size = 2



def draw_msra_gaussian(heatmap, center, sigma=2):
    tmp_size = sigma * 6
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap

    size = 2 * tmp_size + 1
    
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
    g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
    return heatmap

def draw_dense_reg(regmap, heatmap, center, value, radius, is_offset=False):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    value = np.array(value, dtype=np.float32).reshape(-1, 1, 1)
    dim = value.shape[0]
    reg = np.ones((dim, diameter*2+1, diameter*2+1), dtype=np.float32) * value
    if is_offset and dim == 2:
        delta = np.arange(diameter*2+1) - radius
        reg[0] = reg[0] - delta.reshape(1, -1)
        reg[1] = reg[1] - delta.reshape(-1, 1)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]
    
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_regmap = regmap[:, y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom,
                             radius - left:radius + right]
    masked_reg = reg[:, radius - top:radius + bottom,
                              radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        idx = (masked_gaussian >= masked_heatmap).reshape(
          1, masked_gaussian.shape[0], masked_gaussian.shape[1])
    masked_regmap = (1-idx) * masked_regmap + idx * masked_reg
    regmap[:, y - top:y + bottom, x - left:x + right] = masked_regmap
    return regmap

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def make_hm_regr(target, target_shape):
    hm_g = torch.zeros([target.shape[0], input_size//MODEL_SCALE, input_size//MODEL_SCALE])

    regr_g = torch.zeros([target.shape[0], 2, input_size//MODEL_SCALE, input_size//MODEL_SCALE])
    

    
    for k in range(target.shape[0]):
        loc_target = target[k, :target_shape[k][0]].numpy()
        hm = np.zeros([input_size//MODEL_SCALE, input_size//MODEL_SCALE])
        regr = np.zeros([2, input_size//MODEL_SCALE, input_size//MODEL_SCALE])

        if len(target) == 0:
            return hm, regr

        center = np.array([(loc_target[:,0]+loc_target[:,2])//2, (loc_target[:,1]+loc_target[:,3])//2, 
                       (loc_target[:,2]+loc_target[:,0]), (loc_target[:,3]+loc_target[:,1])
                      ]).T

        for c in center:
            hm = draw_msra_gaussian(hm, [int(c[0])//MODEL_SCALE//IN_SCALE, int(c[1])//MODEL_SCALE//IN_SCALE], 
                                    sigma=np.clip(c[2]*c[3]//2000, 2, 4))    

        # convert targets to its center.
        regrs = center[:, 2:]/input_size/IN_SCALE

        # plot regr values to mask
        for r, c in zip(regrs, center):
            for i in range(-2, 3):
                for j in range(-2, 3):
                    try:
                        regr[:, int(c[0])//MODEL_SCALE//IN_SCALE+i, 
                             int(c[1])//MODEL_SCALE//IN_SCALE+j] = r
                    except:
                        pass
        regr[0] = regr[0].T
        regr[1] = regr[1].T

        hm_g[k] =  torch.tensor(hm)
        regr_g[k] =  torch.tensor(regr)
    
    
    return hm_g, regr_g


def pred2box(hm, regr, thresh=0.99):
    # make binding box from heatmaps
    # thresh: threshold for logits.
        
    # get center
    pred = hm > thresh
    pred_center = np.where(hm>thresh)
    # get regressions
    pred_r = regr[:,pred].T

    # wrap as boxes
    # [xmin, ymin, xmax, ymax]
    # size as original image.
    boxes = []
    scores = hm[pred]
    for i, b in enumerate(pred_r):
        arr = np.array([pred_center[1][i]*MODEL_SCALE-np.abs(b[0]*input_size//2), 
                        pred_center[0][i]*MODEL_SCALE-np.abs(b[1]*input_size//2), 
                        pred_center[1][i]*MODEL_SCALE+np.abs(b[0]*input_size//2), 
                        pred_center[0][i]*MODEL_SCALE+np.abs(b[1]*input_size//2)])
        arr = np.clip(arr, 0, input_size)

        boxes.append(arr)
    return np.asarray(boxes), scores

def pred2centers(hm, regr, thresh=0.99):
    # make binding box from heatmaps
    # thresh: threshold for logits.
        
    # get center
    pred = hm > thresh
    pred_center = np.where(hm>thresh)
    # get regressions
    pred_r = regr[:,pred].T

    # wrap as boxes
    # [xmin, ymin, width, height]
    # size as original image.
    scores = hm[pred]
    
    return scores, pred_center

def get_true_centers(pred, scores, dist = 10):
    x = pred[1]
    y = pred[0]

    # Объединяем координаты и сортируем точки по убыванию их scores
    indices = np.argsort(-scores)
    sorted_points = np.column_stack((x[indices], y[indices]))
    sorted_scores = scores[indices]

    # Порог расстояния для фильтрации близких точек
    distance_threshold = dist

    # Маска для отслеживания, какие точки остаются
    keep = np.ones(len(scores), dtype=bool)

    for i in range(len(sorted_points)):
        if not keep[indices[i]]:
            continue
        # Вычисляем расстояния от текущей точки до всех остальных
        distances = np.sqrt(np.sum((sorted_points[i] - sorted_points)**2, axis=1))
        # Отфильтровываем точки, которые слишком близко и имеют меньший score
        for j in range(i + 1, len(sorted_points)):
            if distances[j] < distance_threshold and sorted_scores[j] < sorted_scores[i]:
                keep[indices[j]] = False

    # Результат: фильтрованный список точек
    filtered_points = sorted_points[keep[indices]]
    filtered_scores = sorted_scores[keep[indices]]
    return filtered_points, filtered_scores

def pool(data):
    stride = 3
    for y in np.arange(1,data.shape[1]-1, stride):
        for x in np.arange(1, data.shape[0]-1, stride):
            a_2d = data[x-1:x+2, y-1:y+2]
            max = np.asarray(np.unravel_index(np.argmax(a_2d), a_2d.shape))            
            for c1 in range(3):
                for c2 in range(3):
                    #print(c1,c2)
                    if not (c1== max[0] and c2 == max[1]):
                        data[x+c1-1, y+c2-1] = -1
    return data


def showbox(img, hm, regr, thresh=0.9):
    boxes, _ = pred2box(hm, regr, thresh=thresh)
    print("preds:",boxes.shape)
    sample = img

    for box in boxes:
        # upper-left, lower-right
        cv2.rectangle(sample,
                      (int(box[0]), int(box[1]+box[3])),
                      (int(box[0]+box[2]), int(box[1])),
                      (220, 0, 0), 3)
    return sample

def showgtbox(img, hm, regr, thresh=0.9):
    boxes, _ = pred2box(hm, regr, thresh=thresh)
    print("GT boxes:", boxes.shape)
    sample = img

    for box in boxes:
        cv2.rectangle(sample,
                      (int(box[0]), int(box[1]+box[3])),
                      (int(box[0]+box[2]), int(box[1])),
                      (0, 220, 0), 3)
    return sample


def calculate_accuracy_metrics(predicted_points, true_points, threshold=3):
    # Считаем TP, FP, FN
    tp = 0
    fp = 0
    fn = 0

    # Используем флаги для отслеживания найденных истинных точек
    matched = np.zeros(len(true_points), dtype=bool)

    for pred in predicted_points:
        # Находим расстояние от предсказанной точки до всех истинных точек
        distances = np.sqrt(((true_points - pred) ** 2).sum(axis=1))
        
        # Проверяем, находится ли предсказанная точка в пределах порога от какой-либо истинной точки
        if np.any(distances < threshold):
            tp += 1
            matched[np.argmin(distances)] = True
        else:
            fp += 1
    
    fn = len(true_points) - np.sum(matched)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1


def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def evaluate_keypoints(true_points, pred_points, radius):
    TP = 0
    FP = 0
    FN = 0
    
    # Флаги для отслеживания сопоставленных истинных точек
    matched_true_points = [False] * len(true_points)
    
    # Проверяем каждую предсказанную точку
    for pred in pred_points:
        found_match = False
        for i, true in enumerate(true_points):
            try:
                if calculate_distance(pred, true) <= radius:
                    if not matched_true_points[i]:  
                        TP += 1
                        matched_true_points[i] = True
                        found_match = True
                        break
            except:
                continue
        if not found_match:
            FP += 1

    FN = matched_true_points.count(False)
    
    return TP, FP, FN