"""Model Trainer

author: Masahiro Hayashi

This script defines custom loss functions for image segmentation, which
includes Dice Loss and Weighted Cross Entropy Loss.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Function


def dice_loss(pred, target, smooth=1.):
    """Dice loss
    """
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()


class Weighted_Cross_Entropy_Loss(torch.nn.Module):
    """Cross entropy loss that uses weight maps."""

    def __init__(self, w_1 = 50, w_2 = 1):
        super(Weighted_Cross_Entropy_Loss, self).__init__()
        self.w_1 = w_1
        self.w_2 = w_2

    def forward(self, pred, target):#, weights
        n, c, H, W = pred.shape
        # # Calculate log probabilities
        logp = F.log_softmax(pred, dim=1)
        

        # Gather log probabilities with respect to target
        logp = torch.gather(logp, 1, target.view(n, 1, H, W))

        w_1 = self.w_1#H * W / (1+int(target.sum()))#60
        w_2 = self.w_2
        
        weights = (target * (w_1 - w_2) + w_2)/(w_1 + w_2)
        # Multiply with weights
        weighted_logp = (logp * weights).view(n, -1)

        # Rescale so that loss is in approx. same interval
        weighted_loss = weighted_logp.sum(1) / weights.view(n, -1).sum(1)

        # Average over mini-batch
        weighted_loss = -weighted_loss.mean()

        return weighted_loss
    
class OAN_Focal_Loss(torch.nn.Module):
    """Cross entropy loss that uses weight maps."""

    def __init__(self, alpha = 0.25, gamma = 2):
        super(OAN_Focal_Loss, self).__init__()
        
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):#, weights
        
        
        n, c, H, W = pred.shape
        
        soft_p = F.softmax(pred, dim=1)
        
        P = torch.gather(soft_p, 1, target.view(n, 1, H, W))

        log = self.alpha * (1 - P) ** self.gamma * torch.log(P + 1e-6)

        # Average over mini-batch
        log = -log.mean()

        return log
    
class IOU_loss(torch.nn.Module):
    """Cross entropy loss that uses weight maps."""

    def __init__(self):
        super(IOU_loss, self).__init__()

    def forward(self, pred, target, smooth=1e-6):
        probs = F.softmax(pred, dim=1)
        

        intersection = (probs * target).sum(dim=[2, 3])
        union = probs.sum(dim=[2, 3]) + target.sum(dim=[2, 3]) - intersection

        IoU = (intersection + smooth) / (union + smooth)
        
        return 1 - IoU.mean() 
    
class MSE(torch.nn.Module):
    """Cross entropy loss that uses weight maps."""

    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, target):
        
        return F.mse_loss(pred, target, reduction='mean')


class MSE_CE(torch.nn.Module):
    """Cross entropy loss that uses weight maps."""

    def __init__(self, alpha = 50, betta = 1):
        super(MSE_CE, self).__init__()
        self.alpha = alpha
        self.betta = betta
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):#, weights
        n, c, H, W = pred.shape
        
        p_pred = torch.sigmoid(pred)
        
        mse_loss = F.mse_loss(p_pred * 10, target * 10, reduction='mean')
        
        ce_loss = self.criterion(pred, target)

        return mse_loss, ce_loss
    
class MSE_WCE(torch.nn.Module):
    """Cross entropy loss that uses weight maps."""

    def __init__(self, w = 10):
        super(MSE_WCE, self).__init__()
        self.w = w
        

    def forward(self, pred, target):#, weights
        n, c, H, W = pred.shape
        
        
        pos_weight = target * self.w
        criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight)
        
        p_pred = torch.sigmoid(pred)
        
        mse_loss = F.mse_loss(p_pred * 10, target * 10, reduction='mean')
        
        ce_loss = criterion(pred, target)

        return mse_loss, ce_loss
    
    
class MSE_WCE_IOU(torch.nn.Module):
    """Cross entropy loss that uses weight maps."""

    def __init__(self, w = 10):
        super(MSE_WCE_IOU, self).__init__()
        self.w = w
        

    def forward(self, pred, target):#, weights
        n, c, H, W = pred.shape
        
        
        pos_weight = target * self.w
        criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight)
        
        p_pred = torch.sigmoid(pred)
        
        mse_loss = F.mse_loss(p_pred * 10, target * 10, reduction='mean')
        
        ce_loss = criterion(pred, target)
        
        eps = 1e-6
        pred_flat = p_pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        
        union = pred_flat.sum() + target_flat.sum() - intersection
        
        iou = (intersection + eps) / (union + eps)

        return mse_loss, ce_loss, 1 - iou


class MSE_IOU(torch.nn.Module):
    """Cross entropy loss that uses weight maps."""

    def __init__(self, w = 10):
        super(MSE_IOU, self).__init__()
        self.w = w
        

    def forward(self, pred, target):#, weights
        n, c, H, W = pred.shape
        
        
        p_pred = torch.sigmoid(pred)
        
        mse_loss = F.mse_loss(p_pred * 10, target * 10, reduction='mean')
        
        
        eps = 1e-6
        pred_flat = p_pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        
        union = pred_flat.sum() + target_flat.sum() - intersection
        
        iou = (intersection + eps) / (union + eps)

        return mse_loss, 1 - iou