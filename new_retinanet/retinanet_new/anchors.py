import numpy as np
import torch
import torch.nn as nn
import time


class Anchors(nn.Module):
    def __init__(self, pyramid_levels=[3, 4, 5, 6, 7], strides=None, sizes=None, ratios=None, scales=None):
        super(Anchors, self).__init__()
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.pyramid_levels = pyramid_levels
        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]
        if sizes is None:
            self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]
        if scales is None:
            self.scales = torch.tensor([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)],requires_grad = False, device=self.device)
        if ratios is None:
            self.ratios = torch.tensor([0.5, 1, 2],requires_grad = False, device=self.device)
            self.repeat_ratios = torch.repeat_interleave(self.ratios, len(self.scales))
        
        
            
        

    def forward(self, image):
        # self.pyramid_levels = [3, 4, 5]
        # self.strides = [2 ** x for x in self.pyramid_levels]
        # self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]
        # t = time.time()
        
        # self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # self.ratios = torch.tensor([0.5, 1, 2],requires_grad = False, device=self.device)
        # self.scales = torch.tensor([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)],requires_grad = False, device=self.device)
        # self.flag = True
        

        # print(f'anchor t: {t1 - t}')

        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]
        

        
        num_anchors = len(self.ratios) * len(self.scales)
        all_anchors_len = 0
        all_anchors_offset_list = [0]
        for shapes in image_shapes:
            all_anchors_len += (shapes[0] * shapes[1]) * num_anchors
            all_anchors_offset_list.append(all_anchors_len)

        all_anchors = torch.zeros(all_anchors_len,4, requires_grad = False, device = self.device).float()
        
        anchors = torch.zeros((num_anchors, 4), requires_grad = False, device=self.device)
        # print(f'self.ratios.device {self.ratios.device}')


        for idx, p in enumerate(self.pyramid_levels):
            
            anchors         = generate_anchors(anchors, base_size=self.sizes[idx], repeat_ratios = self.repeat_ratios, ratios=self.ratios,  scales=self.scales, device = self.device)

            all_anchors[all_anchors_offset_list[idx] : all_anchors_offset_list[idx+1], :] = shift(image_shapes[idx], self.strides[idx], anchors, device = self.device)

            

        all_anchors = all_anchors.unsqueeze(0)
        return all_anchors
    
    

def generate_anchors(anchors, base_size=16, repeat_ratios=None, ratios=None, scales=None, device = torch.device('cpu')):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    
    if ratios is None:
        ratios = torch.tensor([0.5, 1, 2],requires_grad = False, device=device)

    if scales is None:
        scales = torch.tensor([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)],requires_grad = False, device=device)

    num_anchors = len(ratios) * len(scales)


    # initialize output anchors
    anchors = anchors * 0

    # scale base_size
    anchors[:, 2:] = base_size * torch.tile(scales, (2, len(ratios))).T
    


    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]
    

    torch_sqrt = torch.sqrt(areas / repeat_ratios)

    anchors[:, 2] = torch_sqrt

    anchors[:, 3] = anchors[:, 2] * repeat_ratios #torch.repeat_interleave(ratios, len(scales))
    


    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= torch.tile(anchors[:, 2] * 0.5, (2, 1)).T

    anchors[:, 1::2] -= torch.tile(anchors[:, 3] * 0.5, (2, 1)).T
    


    return anchors

def compute_shape(image_shape, pyramid_levels):
    """Compute shapes based on pyramid levels.

    :param image_shape:
    :param pyramid_levels:
    :return:
    """
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes


def anchors_for_shape(
    image_shape,
    pyramid_levels=None,
    ratios=None,
    scales=None,
    strides=None,
    sizes=None,
    shapes_callback=None,
):

    image_shapes = compute_shape(image_shape, pyramid_levels)

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4))
    for idx, p in enumerate(pyramid_levels):
        anchors         = generate_anchors(base_size=sizes[idx], ratios=ratios, scales=scales)
        shifted_anchors = shift(image_shapes[idx], strides[idx], anchors)
        all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)

    return all_anchors


def shift(shape, stride, anchors, device = torch.device('cpu')):
    
    shift_x = (torch.arange(0, shape[1],requires_grad = False, device=device) + 0.5) * stride
    shift_y = (torch.arange(0, shape[0],requires_grad = False, device=device) + 0.5) * stride

    shift_x, shift_y = torch.meshgrid(shift_x, shift_y, indexing='xy')

    
    shift_x = shift_x.contiguous().view(-1)  # Предполагая, что shift_x это одномерный тензор
    shift_y = shift_y.contiguous().view(-1)  # Аналогично для shift_y

    shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=0).T


    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.size(0)
    K = shifts.size(0)


    all_anchors = (anchors.view(1, A, 4) + shifts.unsqueeze(0).permute(1,0,2)).reshape(K * A, 4)
    

    

    return all_anchors

