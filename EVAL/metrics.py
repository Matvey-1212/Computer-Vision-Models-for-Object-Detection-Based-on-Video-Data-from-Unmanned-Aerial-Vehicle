import numpy as np
from copy import copy
import torch


def compute_overlap(a: np.array, b: np.array) -> np.array:
    """
    Args
        a: (N, 4) ndarray of float [xmin, ymin, xmax, ymax]
        b: (K, 4) ndarray of float [xmin, ymin, xmax, ymax]

    Returns
        overlaps: (N, K) ndarray of overlap between boxes a and boxes b
    """
    a_area = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    b_area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    dx = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], axis=1), b[:, 0])
    dy = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], axis=1), b[:, 1])

    intersection = np.maximum(dx, 0) * np.maximum(dy, 0)
    union = np.expand_dims(a_area, axis=1) + b_area - intersection
    overlaps = intersection / union

    return overlaps


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evaluate_f_score(precision, recall, beta=1.0):
    f_score = (1 + beta ** 2) * precision * recall / np.fmax(precision * (beta ** 2) + recall,
                                                      np.finfo(np.float64).eps)

    return f_score


def evaluate_map(inference_res, iou_threshold=0.5, score_threshold=0.05, num_cls=1, beta=0.5):
    """
    # Arguments
        inference_res   : inference results for whole imageset List((target,prediction)),
            where targets {'boxes':np.array[4,n], 'labels':np.array[n]},
            prediction {'boxes':np.array[4,n], 'labels':np.array[n], scores: np.array[n]}
            example:

            [({'boxes': np.array([[1321.8750,  274.6667, 1348.8750,  312.6667]]),
                'labels': np.array([1])},
              {'boxes': np.array([[1323.5446,  275.2711, 1350.2203,  315.9069],
                        [ 119.2671, 1227.5459,  171.1528, 1277.9830],
                        [ 240.5078, 1147.3656,  270.7879, 1205.0126],
                        [ 140.9097, 1231.9814,  173.9967, 1285.4724]]),
                'scores': np.array([0.9568, 0.3488, 0.1418, 0.0771]),
                'labels': np.array([1, 1, 1, 1])}),
             ({'boxes': np.array([[ 798.7500, 1357.3334,  837.7500, 1396.6666],
                        [ 829.1250,  777.3333,  873.3750,  818.0000],
                        [ 886.5000,   34.6667,  916.5000,   77.3333]]),
                'labels': np.array([1, 1, 1])},
              {'boxes': np.array([[ 796.5808, 1354.9255,  836.5349, 1395.8972],
                        [ 828.8597,  777.9426,  872.5923,  819.8660],
                        [ 887.7839,   37.1435,  914.8092,   76.3933]]),
                'scores': np.array([0.9452, 0.8701, 0.8424]),
                'labels': np.array([1, 1, 1])})]

        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        num_cls: The amount of classes in dataset.
        return: mAP, mF05 per classes
    """
    per_cls_results = {}
    for cls in range(num_cls):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0
        num_detections = 0

        for i, (t, p) in enumerate(inference_res):
            annotations = copy(t)
            detections = copy(p)
            detected_annotations = []
            cls_annot_idx = annotations['labels'] == cls
            cls_detect_idx = detections['labels'] == cls

            annotations['boxes'] = annotations['boxes'][cls_annot_idx]
            annotations['labels'] = annotations['labels'][cls_annot_idx]
            num_annotations += len(annotations['labels'])

            detections['boxes'] = detections['boxes'][cls_detect_idx]
            detections['labels'] = detections['labels'][cls_detect_idx]
            detections['scores'] = detections['scores'][cls_detect_idx]
            num_detections += len(detections['labels'])

            if annotations['labels'].shape[0] == 0:  # no objects was there
                false_positives = np.append(false_positives, np.ones(detections['labels'].shape[0]))
                true_positives = np.append(true_positives, np.zeros(detections['labels'].shape[0]))
                continue

            for d in np.arange(detections['labels'].shape[0]):
                if detections['scores'][d] > score_threshold:
                    scores = np.append(scores, detections['scores'][d])

                    overlaps = compute_overlap(np.expand_dims(detections['boxes'][d].astype(np.double), axis=0),
                                               annotations['boxes'].astype(np.double))
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap = overlaps[0, assigned_annotation][0]

                    if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)

        if num_annotations == 0 and num_detections == 0:
            per_cls_results[cls] = [1., 1.]
            continue

        # F_score@IoU
        plain_recall = np.sum(true_positives) / np.fmax(num_annotations, np.finfo(np.float64).eps)
        plain_precision = np.sum(true_positives) / np.fmax(np.sum(true_positives) + np.sum(false_positives),
                                                           np.finfo(np.float64).eps)

        f_score = evaluate_f_score(plain_precision, plain_recall, beta=beta)

        # compute false positives and true positives
        indices = np.argsort(scores)[::-1]
        false_positives = np.cumsum(false_positives[indices])
        true_positives = np.cumsum(true_positives[indices])
        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.fmax(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = compute_ap(recall, precision)

        per_cls_results[cls] = [average_precision, f_score]

    map = np.mean([per_cls_results[cls][0] for cls in range(num_cls)])
    mFscore = np.mean([per_cls_results[cls][1] for cls in range(num_cls)])

    return map, mFscore


def evaluate(inference_res, iou_thr_min=0.5, iou_thr_max=0.95, iou_thr_step=0.05, score_threshold=0.05,
             num_cls=1, beta=0.5):

    map = []
    mFscore = []
    for thr in np.linspace(iou_thr_min, iou_thr_max, int((iou_thr_max - iou_thr_min) / iou_thr_step)):
        _map, _mFscore = evaluate_map(inference_res, thr, score_threshold=score_threshold, num_cls=num_cls, beta=beta)
        map.append(_map)
        mFscore.append(_mFscore)

    return np.mean(map), np.mean(mFscore)

def calculate_semantic_metrics(outputs, labels):
    preds = outputs > 0.5
    TP = (preds & labels).sum().float()
    FP = ((preds == 1) & (labels == 0)).sum().float()
    TN = ((preds == 0) & (labels == 0)).sum().float()
    FN = ((preds == 0) & (labels == 1)).sum().float()

    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else torch.tensor(0.0)
    recall = TP / (TP + FN) if (TP + FN) > 0 else torch.tensor(0.0)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else torch.tensor(0.0)

    return accuracy.item(), precision.item(), recall.item(), f1.item()

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):

    intersection = (outputs & labels).float().sum((-2, -1))  

    union = (outputs | labels).float().sum((-2, -1))   
    
    iou = (intersection + 1e-6) / (union + 1e-6)  
    

    iou_mean = iou.mean()  
    return iou_mean

def dice_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    intersection = (outputs & labels).float().sum((-2, -1))
    dice = (2. * intersection + 1e-6) / (outputs.float().sum((-2, -1)) + labels.float().sum((-2, -1)) + 1e-6)
    
    dice_mean = dice.mean() 
    return dice_mean
