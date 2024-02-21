"""Model Trainer

author: Masahiro Hayashi

This script defines custom loss functions for image segmentation, which
includes Dice Loss and Weighted Cross Entropy Loss.
"""

import torch
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

    def __init__(self):
        super(Weighted_Cross_Entropy_Loss, self).__init__()

    def forward(self, pred, target):#, weights
        n, c, H, W = pred.shape
        # # Calculate log probabilities
        logp = F.log_softmax(pred, dim=1)
        

        # Gather log probabilities with respect to target
        logp = torch.gather(logp, 1, target.view(n, 1, H, W))

        w_1 = 50#H * W / (1+int(target.sum()))#60
        w_2 = 1
        
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

    def __init__(self):
        super(OAN_Focal_Loss, self).__init__()

    def forward(self, pred, target, annot):#, weights
        
        print(f'pred {pred.shape}')
        print(f'annot {annot.shape}')
        print(f'target {target.shape}')
        exit()
        
        n, c, H, W = pred.shape
        # # Calculate log probabilities
        logp = F.log_softmax(pred, dim=1)
        

        # Gather log probabilities with respect to target
        logp = torch.gather(logp, 1, target.view(n, 1, H, W))

        w_1 = 50#H * W / (1+int(target.sum()))
        w_2 = 1
        
        weights = (target * (w_1 - w_2) + w_2)/(w_1 + w_2)
        # Multiply with weights
        weighted_logp = (logp * weights).view(n, -1)

        # Rescale so that loss is in approx. same interval
        weighted_loss = weighted_logp.sum(1) / weights.view(n, -1).sum(1)

        # Average over mini-batch
        weighted_loss = -weighted_loss.mean()

        return weighted_loss
    
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

# def class_weight(target):
#     weight = torch.zeros(batch_size, H, W)
#     for i in range(out_channels):
#         i_t = i * torch.ones([batch_size, H, W], dtype=torch.long)
#         loc_i = (target == i_t).to(torch.long)
#         count_i = loc_i.view(out_channels, -1).sum(1)
#         total = H*W
#         weight_i = total / count_i
#         weight_t = loc_i * weight_i.view(-1, 1, 1)
#         weight += weight_t
#     return weight
