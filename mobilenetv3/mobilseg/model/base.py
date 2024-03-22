"""Base Model for Semantic Segmentation"""
import torch.nn as nn
import torch

from .base_model import mobilenet_v3_large_1_0, mobilenet_v3_small_1_0

__all__ = ['SegBaseModel']

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

class SegBaseModel(nn.Module):
    def __init__(self, nclass, aux=False, backbone='mobilenetv3_small', pretrained_base=True, attention = False, **kwargs):
        super(SegBaseModel, self).__init__()
        self.nclass = nclass
        self.aux = aux
        self.mode = backbone.split('_')[-1]
        self.attention = attention
        assert self.mode in ['large', 'small']
        if backbone == 'mobilenetv3_large':
            self.pretrained = mobilenet_v3_large_1_0(dilated=True, pretrained=pretrained_base, attention = attention, **kwargs)
            
            self.a1 = Attention_block(F_g=112,F_l=24,F_int=56)
            self.Maxpool1 = nn.AvgPool2d(kernel_size=4,stride=4)
            
        elif backbone == 'mobilenetv3_small':
            self.pretrained = mobilenet_v3_small_1_0(dilated=True, pretrained=pretrained_base, attention = attention, **kwargs)
            
            self.a1 = Attention_block(F_g=48,F_l=16,F_int=24)
            self.Maxpool1 = nn.AvgPool2d(kernel_size=4,stride=4)
            
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        
        
        
        


    def base_forward(self, x):
        """forwarding pre-trained network"""
        x = self.pretrained.conv1(x)

        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        
        if self.attention:
            c1_at = self.a1(g = c3, x = self.Maxpool1(c1))        #added
            c3 = torch.cat((c3, c1_at), dim=1)     #added
        
        c4 = self.pretrained.layer4(c3)
        c4 = self.pretrained.conv5(c4)
        
        # x1 = self.pretrained.oan_layer1(c4)
        # x2 = self.pretrained.oan_layer2(x1)
        # x3 = self.pretrained.oan_layer3(x2)
        
        # print(f'c1 {c1.shape}')
        # print(f'c2 {c2.shape}')
        # print(f'c3 {c3.shape}')
        # print(f'c4 {c4.shape}')
        # print(f'x1 {x1.shape}')
        # print(f'x2 {x2.shape}')
        # print(f'x3 {x3.shape}')


        return c1, c2, c3, c4#, x3


if __name__ == '__main__':
    model = SegBaseModel(20, pretrained_base=False)
