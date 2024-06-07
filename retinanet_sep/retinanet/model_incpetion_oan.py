import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
from torchvision.ops import nms
from retinanet.utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from retinanet.anchors import Anchors
from retinanet import losses
from retinanet import OAN
import torch.nn.functional as F
from retinanet.aploss import APLoss

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1
        
class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1

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
    
class CustomPyramidFeaturesAT2_newLayer(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, layers = [3,4,5,6,7], feature_size=256):
        super(CustomPyramidFeaturesAT2_newLayer, self).__init__()
        
        self.layers = layers

        # upsample C5 to get P5 from the FPN paper
        self.P5_1_1 = nn.Conv2d(C5_size, feature_size//2, kernel_size=1, stride=1, padding=0)
        self.P5_1_3 = nn.Conv2d(C5_size, feature_size//2, kernel_size=3, stride=1, padding=1)
        self.P5_1_5 = nn.Conv2d(C5_size, feature_size//2, kernel_size=3, stride=1, padding=2, dilation=2)
        
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size//2, feature_size, kernel_size=3, stride=1, padding=1)
        

        
        self.P4_1_1 = nn.Conv2d(C4_size, feature_size//2, kernel_size=1, stride=1, padding=0)
        self.P4_1_3 = nn.Conv2d(C4_size, feature_size//2, kernel_size=3, stride=1, padding=1)
        self.P4_1_5 = nn.Conv2d(C4_size, feature_size//2, kernel_size=3, stride=1, padding=2, dilation=2)
        self.Att4 = Attention_block(F_g=128,F_l=128,F_int=128)

        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        

        
        self.P3_1_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_1_3 = nn.Conv2d(C3_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P3_1_5 = nn.Conv2d(C3_size, feature_size, kernel_size=3, stride=1, padding=2, dilation=2)
        self.Att3 = Attention_block(F_g=256,F_l=256,F_int=256)

        
        self.P3_2 = nn.Conv2d(feature_size*2, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)


    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1_1(C5) #
        P5_x = P5_x + self.P5_1_3(C5) 
        P5_x = P5_x + self.P5_1_5(C5) 
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1_1(C4) #
        P4_x = P4_x + self.P4_1_3(C4) 
        P4_x = P4_x + self.P4_1_5(C4) 
        P4_x = self.Att4(g=P5_upsampled_x, x=P4_x)
        P4_x = torch.cat((P5_upsampled_x,P4_x),dim=1) #P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1_1(C3) #
        P3_x = P3_x + self.P3_1_3(C3) 
        P3_x = P3_x + self.P3_1_5(C3) 
        P3_x = self.Att3(g=P4_upsampled_x, x=P3_x)
        P3_x = torch.cat((P4_upsampled_x,P3_x),dim=1) #P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        
        return [P3_x, P4_x, P5_x, P6_x, P7_x]

    
class CustomPyramidFeaturesATR2(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, layers = [3,4,5,6,7], feature_size=256):
        super(CustomPyramidFeaturesATR2, self).__init__()
        
        self.layers = layers

        # upsample C5 to get P5 from the FPN paper
        # self.P5_1_1 = nn.Conv2d(C5_size, feature_size//2, kernel_size=1, stride=1, padding=0)
        # self.P5_1_3 = nn.Conv2d(C5_size, feature_size//2, kernel_size=3, stride=1, padding=1)
        # self.P5_1_5 = nn.Conv2d(C5_size, feature_size//2, kernel_size=3, stride=1, padding=2, dilation=2)
        self.RRCNN1 = RRCNN_block(ch_in=C5_size, ch_out=feature_size//2, t=2)
        
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

        self.P5_2 = nn.Conv2d(feature_size//2, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        # self.P4_1_1 = nn.Conv2d(C4_size, feature_size//2, kernel_size=1, stride=1, padding=0)
        # self.P4_1_3 = nn.Conv2d(C4_size, feature_size//2, kernel_size=3, stride=1, padding=1)
        # self.P4_1_5 = nn.Conv2d(C4_size, feature_size//2, kernel_size=3, stride=1, padding=2, dilation=2)
        self.RRCNN2 = RRCNN_block(ch_in=C4_size, ch_out=feature_size//2, t=2)
        
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        # self.P3_1_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        # self.P3_1_3 = nn.Conv2d(C3_size, feature_size, kernel_size=3, stride=1, padding=1)
        # self.P3_1_5 = nn.Conv2d(C3_size, feature_size, kernel_size=3, stride=1, padding=2, dilation=2)
        self.RRCNN3 = RRCNN_block(ch_in=C3_size,ch_out=feature_size,t=2)
        
        self.P3_2 = nn.Conv2d(feature_size * 2, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)


# torch.cat((x3,d4),dim=1)
    def forward(self, inputs):
        C3, C4, C5 = inputs

        # P5_x = self.P5_1_1(C5) + self.P5_1_3(C5) + self.P5_1_5(C5) 
        P5_x = self.RRCNN1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        # P4_x = self.P4_1_1(C4) + self.P4_1_3(C4) + self.P4_1_5(C4) 
        P4_x = self.RRCNN2(C4)
        P4_x = torch.cat((P5_upsampled_x,P4_x),dim=1) #P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        # P3_x = self.P3_1_1(C3) + self.P3_1_3(C3) + self.P3_1_5(C3) 
        P3_x = self.RRCNN3(C3)
        P3_x = torch.cat((P4_upsampled_x,P3_x),dim=1) #P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        
        return [P3_x, P4_x, P5_x, P6_x, P7_x]
    
class CustomPyramidFeaturesR2(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, layers = [3,4,5,6,7], feature_size=256):
        super(CustomPyramidFeaturesR2, self).__init__()
        
        self.layers = layers

        # upsample C5 to get P5 from the FPN paper
        self.RRCNN1 = RRCNN_block(ch_in=C5_size, ch_out=feature_size//2, t=2)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size//2, feature_size, kernel_size=3, stride=1, padding=1)
        

        self.RRCNN2 = RRCNN_block(ch_in=C4_size, ch_out=feature_size//2, t=2)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        

        self.RRCNN3 = RRCNN_block(ch_in=C3_size,ch_out=feature_size,t=2)
        self.P3_2 = nn.Conv2d(feature_size*2, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)


    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.RRCNN1(C5) # P5_x = self.P5_1_1(C5) #+ self.P5_1_3(C5) + self.P5_1_5(C5) 
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.RRCNN2(C4) # P4_x = self.P4_1_1(C4) #+ self.P4_1_3(C4) + self.P4_1_5(C4) 
        # P4_x = self.Att4(g=P5_upsampled_x, x=P4_x)
        P4_x = torch.cat((P5_upsampled_x,P4_x),dim=1) #P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.RRCNN3(C3) # P3_x = self.P3_1_1(C3) #+ self.P3_1_3(C3) + self.P3_1_5(C3) 
        # P3_x = self.Att3(g=P4_upsampled_x, x=P3_x)
        P3_x = torch.cat((P4_upsampled_x,P3_x),dim=1) #P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        
        return [P3_x, P4_x, P5_x, P6_x, P7_x]
    
    
class CustomPyramidFeaturesAT2_newLayer_P2(nn.Module):
    def __init__(self, C2_size, C3_size, C4_size, C5_size, layers = [3,4,5,6,7], feature_size=256):
        super(CustomPyramidFeaturesAT2_newLayer_P2, self).__init__()
        
        self.layers = layers

        # upsample C5 to get P5 from the FPN paper
        self.P5_1_1 = nn.Conv2d(C5_size, feature_size//2, kernel_size=1, stride=1, padding=0)
        self.P5_1_3 = nn.Conv2d(C5_size, feature_size//2, kernel_size=3, stride=1, padding=1)
        self.P5_1_5 = nn.Conv2d(C5_size, feature_size//2, kernel_size=3, stride=1, padding=2, dilation=2)
        
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size//2, feature_size, kernel_size=3, stride=1, padding=1)
        

        
        self.P4_1_1 = nn.Conv2d(C4_size, feature_size//2, kernel_size=1, stride=1, padding=0)
        self.P4_1_3 = nn.Conv2d(C4_size, feature_size//2, kernel_size=3, stride=1, padding=1)
        self.P4_1_5 = nn.Conv2d(C4_size, feature_size//2, kernel_size=3, stride=1, padding=2, dilation=2)
        self.Att4 = Attention_block(F_g=128,F_l=128,F_int=128)

        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        

        
        self.P3_1_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_1_3 = nn.Conv2d(C3_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P3_1_5 = nn.Conv2d(C3_size, feature_size, kernel_size=3, stride=1, padding=2, dilation=2)
        self.Att3 = Attention_block(F_g=256,F_l=256,F_int=256)

        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(feature_size*2, feature_size, kernel_size=3, stride=1, padding=1)
        

        self.P2_1_1 = nn.Conv2d(C2_size, feature_size * 2, kernel_size=1, stride=1, padding=0)
        self.P2_1_3 = nn.Conv2d(C2_size, feature_size * 2, kernel_size=3, stride=1, padding=1)
        self.P2_1_5 = nn.Conv2d(C2_size, feature_size * 2, kernel_size=3, stride=1, padding=2, dilation=2)
        self.Att2 = Attention_block(F_g=256 * 2,F_l=256 * 2,F_int=256 * 2)

        self.P2_2 = nn.Conv2d(feature_size*4, feature_size, kernel_size=3, stride=1, padding=1)
        


    def forward(self, inputs):
        C2, C3, C4, C5 = inputs

        P5_x = self.P5_1_1(C5) #
        P5_x = P5_x + self.P5_1_3(C5) 
        P5_x = P5_x + self.P5_1_5(C5) 
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1_1(C4) #
        P4_x = P4_x + self.P4_1_3(C4) 
        P4_x = P4_x + self.P4_1_5(C4) 
        P4_x = self.Att4(g=P5_upsampled_x, x=P4_x)
        P4_x = torch.cat((P5_upsampled_x,P4_x),dim=1) #P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1_1(C3) #
        P3_x = P3_x + self.P3_1_3(C3) 
        P3_x = P3_x + self.P3_1_5(C3) 
        P3_x = self.Att3(g=P4_upsampled_x, x=P3_x)
        P3_x = torch.cat((P4_upsampled_x,P3_x),dim=1) #P3_x + P4_upsampled_x
        P3_upsampled_x = self.P3_upsampled(P3_x)
        P3_x = self.P3_2(P3_x)

        P2_x = self.P2_1_1(C2) #
        P2_x = P2_x + self.P2_1_3(C2) 
        P2_x = P2_x + self.P2_1_5(C2) 
        P2_x = self.Att2(g=P3_upsampled_x, x=P2_x)
        P2_x = torch.cat((P3_upsampled_x,P2_x),dim=1) #P3_x + P4_upsampled_x
        P2_x = self.P2_2(P2_x)
        
             
        return [P2_x, P3_x, P4_x, P5_x]
    
class CustomPyramidFeaturesAT2(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, layers = [3,4,5,6,7], feature_size=256):
        super(CustomPyramidFeaturesAT2, self).__init__()
        
        self.layers = layers

        # upsample C5 to get P5 from the FPN paper
        self.P5_1_1 = nn.Conv2d(C5_size, feature_size//2, kernel_size=1, stride=1, padding=0)
        # self.P5_1_3 = nn.Conv2d(C5_size, feature_size//2, kernel_size=3, stride=1, padding=1)
        # self.P5_1_5 = nn.Conv2d(C5_size, feature_size//2, kernel_size=3, stride=1, padding=2, dilation=2)
        
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size//2, feature_size, kernel_size=3, stride=1, padding=1)
        

        
        self.P4_1_1 = nn.Conv2d(C4_size, feature_size//2, kernel_size=1, stride=1, padding=0)
        # self.P4_1_3 = nn.Conv2d(C4_size, feature_size//2, kernel_size=3, stride=1, padding=1)
        # self.P4_1_5 = nn.Conv2d(C4_size, feature_size//2, kernel_size=3, stride=1, padding=2, dilation=2)
        self.Att4 = Attention_block(F_g=128,F_l=128,F_int=128)

        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        

        
        self.P3_1_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        # self.P3_1_3 = nn.Conv2d(C3_size, feature_size, kernel_size=3, stride=1, padding=1)
        # self.P3_1_5 = nn.Conv2d(C3_size, feature_size, kernel_size=3, stride=1, padding=2, dilation=2)
        self.Att3 = Attention_block(F_g=256,F_l=256,F_int=256)

        
        self.P3_2 = nn.Conv2d(feature_size*2, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)


    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1_1(C5) #+ self.P5_1_3(C5) + self.P5_1_5(C5) 
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1_1(C4) #+ self.P4_1_3(C4) + self.P4_1_5(C4) 
        P4_x = self.Att4(g=P5_upsampled_x, x=P4_x)
        P4_x = torch.cat((P5_upsampled_x,P4_x),dim=1) #P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1_1(C3) #+ self.P3_1_3(C3) + self.P3_1_5(C3) 
        P3_x = self.Att3(g=P4_upsampled_x, x=P3_x)
        P3_x = torch.cat((P4_upsampled_x,P3_x),dim=1) #P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        
        return [P3_x, P4_x, P5_x, P6_x, P7_x]

class CustomPyramidFeaturesAT(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, layers = [3,4,5,6,7], feature_size=256):
        super(CustomPyramidFeaturesAT, self).__init__()
        
        self.layers = layers

        # upsample C5 to get P5 from the FPN paper
        self.P5_1_1 = nn.Conv2d(C5_size, feature_size//2, kernel_size=1, stride=1, padding=0)
        self.P5_1_3 = nn.Conv2d(C5_size, feature_size//2, kernel_size=3, stride=1, padding=1)
        self.P5_1_5 = nn.Conv2d(C5_size, feature_size//2, kernel_size=3, stride=1, padding=2, dilation=2)
        
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

        self.P5_2 = nn.Conv2d(feature_size//2, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1_1 = nn.Conv2d(C4_size, feature_size//2, kernel_size=1, stride=1, padding=0)
        self.P4_1_3 = nn.Conv2d(C4_size, feature_size//2, kernel_size=3, stride=1, padding=1)
        self.P4_1_5 = nn.Conv2d(C4_size, feature_size//2, kernel_size=3, stride=1, padding=2, dilation=2)

        
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_1_3 = nn.Conv2d(C3_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P3_1_5 = nn.Conv2d(C3_size, feature_size, kernel_size=3, stride=1, padding=2, dilation=2)

        
        self.P3_2 = nn.Conv2d(feature_size * 2, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)


# torch.cat((x3,d4),dim=1)
    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1_1(C5) + self.P5_1_3(C5) + self.P5_1_5(C5) 
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1_1(C4) + self.P4_1_3(C4) + self.P4_1_5(C4) 
        P4_x = torch.cat((P5_upsampled_x,P4_x),dim=1) #P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1_1(C3) + self.P3_1_3(C3) + self.P3_1_5(C3) 
        P3_x = torch.cat((P4_upsampled_x,P3_x),dim=1) #P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        
        return [P3_x, P4_x, P5_x, P6_x, P7_x]

class CustomPyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, layers = [3,4,5,6,7], feature_size=256):
        super(CustomPyramidFeatures, self).__init__()
        
        self.layers = layers

        # upsample C5 to get P5 from the FPN paper
        self.P5_1_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_1_3 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P5_1_5 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=1, padding=2, dilation=2)
        
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_1_3 = nn.Conv2d(C4_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P4_1_5 = nn.Conv2d(C4_size, feature_size, kernel_size=3, stride=1, padding=2, dilation=2)
        
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_1_3 = nn.Conv2d(C3_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P3_1_5 = nn.Conv2d(C3_size, feature_size, kernel_size=3, stride=1, padding=2, dilation=2)
        
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1_1(C5) + self.P5_1_3(C5) + self.P5_1_5(C5) 
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1_1(C4) + self.P4_1_3(C4) + self.P4_1_5(C4) 
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1_1(C3) + self.P3_1_3(C3) + self.P3_1_5(C3) 
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        
        return [P3_x, P4_x, P5_x, P6_x, P7_x]

class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class ResNet(nn.Module):

    def __init__(self, num_classes, block, layers, inputs = 3, oan_gamma = 2, oan_alpha = 0.25, loss = 'focal', fpn = PyramidFeatures):
        self.inplanes = 64
        self.num_classes = num_classes
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(inputs, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.oan_layer1 = nn.Conv2d(in_channels=2048,  
                                    out_channels=256,  
                                    kernel_size=3,     
                                    stride=2,          
                                    padding=1) 
        
        self.oan_layer2 = nn.Conv2d(in_channels=256, 
                                    out_channels=512,  
                                    kernel_size=1,     
                                    stride=1,          
                                    padding=0) 
        self.oan_layer3 = nn.Conv2d(in_channels=512,  
                                    out_channels=1,  
                                    kernel_size=1,     
                                    stride=1,          
                                    padding=0) 
        

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        # self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])
        
        # self.fpn = CustomPyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])
        
        # self.fpn = CustomPyramidFeaturesAT(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])
        
        # self.fpn = CustomPyramidFeaturesAT2(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])
        
        # self.fpn = CustomPyramidFeaturesR2(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])
        
        # self.fpn = CustomPyramidFeaturesATR2(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])
        
        self.fpn = fpn(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])
        
        self.anchors = Anchors()
        # self.get_anchor = True
        
        
        

        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=self.num_classes)

        

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        self.focalLoss = losses.FocalLoss()
        
        self.aploss = APLoss.apply

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        # self.freeze_bn() #freeze BatchNorm2d

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):

        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs

        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        x_oan = self.oan_layer1(x4)
        x_oan = self.oan_layer2(x_oan)
        x_oan = self.oan_layer3(x_oan)
        
        # if not self.training:
            

        features = self.fpn([x2, x3, x4])

        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

        
        
        # if self.get_anchor:
        anchors = self.anchors(img_batch)
            # self.get_anchor = False

        if self.training:
            # class_loss1, reg_loss1 = self.focalLoss(classification, regression, anchors, annotations)
            # # class_loss2, reg_loss2 = self.aploss(classification, regression, anchors, annotations)
            
            # # class_loss1, reg_loss1 = self.aploss(classification, regression, anchors, annotations)
            
            # class_loss = class_loss1.mean() #+ class_loss2.mean()
            # reg_loss = reg_loss1.mean() #+ reg_loss2.mean()
            
            return classification, regression, anchors, annotations, x_oan #class_loss, reg_loss#, oan_loss, x_oan#, class_loss_ap, reg_loss_ap
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            finalResult = [[], [], []]

            finalScores = torch.Tensor([])
            finalAnchorBoxesIndexes = torch.Tensor([]).long()
            finalAnchorBoxesCoordinates = torch.Tensor([])

            if torch.cuda.is_available():
                finalScores = finalScores.cuda()
                finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
                finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()

            for i in range(classification.shape[2]):
                scores = torch.squeeze(classification[:, :, i])
                scores_over_thresh = (scores > 0.05)
                if scores_over_thresh.sum() == 0:
                    # no boxes to NMS, just continue
                    continue

                scores = scores[scores_over_thresh]
                anchorBoxes = torch.squeeze(transformed_anchors)
                anchorBoxes = anchorBoxes[scores_over_thresh]
                anchors_nms_idx = nms(anchorBoxes, scores, 0.1)

                finalResult[0].extend(scores[anchors_nms_idx])
                finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
                finalResult[2].extend(anchorBoxes[anchors_nms_idx])

                finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
                finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
                if torch.cuda.is_available():
                    finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

                finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
                finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

            return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates, classification, regression, anchors, x_oan]

class OAN_Retinanet(nn.Module):
    
    def __init__(self, num_classes = 1, inputs = 4):
        super(OAN_Retinanet, self).__init__()
        self.oan = OAN.OAN();
        
        self.retina = ResNet(num_classes, Bottleneck, [3, 8, 36, 3], inputs = inputs)
        
        self.upsample = torch.nn.Upsample(scale_factor=64) 
        
        self.loss = losses.OANFocalLoss(alpha = 0.25, gamma = 2)
        
    def forward(self, inputs):
        
        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs
            
        shape = img_batch.shape
        x = self.oan(img_batch)
        
        score =  self.loss(x.detach(), annotations, (shape[-1], shape[-2]))
        
        x = self.upsample(x)

        x = torch.cat([img_batch, x], dim=1)
        
        
        a, b = self.retina([x,annotations])
        
        
        # output+= score
        return a, b, score

def resnetCustom(num_classes, layers = [3, 4, 6, 3], **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, layers, **kwargs)
    
    return model
        
    

def resnet18(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model


def resnet34(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model


def resnet50(num_classes, pretrained=False, inputs = 3, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], inputs = inputs, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model


def resnet101(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model


def resnet152(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model


    
    
def oan(num_classes, pretrained=False, **kwargs):
    model = OAN_Retinanet()
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model