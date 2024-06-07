import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
from torchvision.ops import nms
from centers.utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from centers.anchors import Anchors
from centers import losses
import torch.nn.functional as F
from centers.aploss import APLoss

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}



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

    


    
class CustomPyramidFeatures_small2_newLayer_P2(nn.Module):
    def __init__(self, C2_size, C3_size, C4_size, C5_size, layers = [3,4,5,6,7], feature_size=256):
        super(CustomPyramidFeatures_small2_newLayer_P2, self).__init__()
        
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
        
        self.P4_12_1 = nn.Conv2d(feature_size, feature_size//2, kernel_size=1, stride=1, padding=0)
        
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        

        
        self.P3_1_1 = nn.Conv2d(C3_size, feature_size//2, kernel_size=1, stride=1, padding=0)
        self.P3_1_3 = nn.Conv2d(C3_size, feature_size//2, kernel_size=3, stride=1, padding=1)
        self.P3_1_5 = nn.Conv2d(C3_size, feature_size//2, kernel_size=3, stride=1, padding=2, dilation=2)
        
        self.P3_12_1 = nn.Conv2d(feature_size, feature_size//2, kernel_size=1, stride=1, padding=0)

        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        

        self.P2_1_1 = nn.Conv2d(C2_size, feature_size//2, kernel_size=1, stride=1, padding=0)
        self.P2_1_3 = nn.Conv2d(C2_size, feature_size//2, kernel_size=3, stride=1, padding=1)
        self.P2_1_5 = nn.Conv2d(C2_size, feature_size//2, kernel_size=3, stride=1, padding=2, dilation=2)

        self.P2_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        


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
        P4_x = torch.cat((P5_upsampled_x,P4_x),dim=1) #P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_12_1(self.P4_upsampled(P4_x))
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1_1(C3) #
        P3_x = P3_x + self.P3_1_3(C3) 
        P3_x = P3_x + self.P3_1_5(C3) 
        P3_x = torch.cat((P4_upsampled_x,P3_x),dim=1) #P3_x + P4_upsampled_x
        P3_upsampled_x = self.P3_12_1(self.P3_upsampled(P3_x))
        P3_x = self.P3_2(P3_x)

        P2_x = self.P2_1_1(C2) #
        P2_x = P2_x + self.P2_1_3(C2) 
        P2_x = P2_x + self.P2_1_5(C2) 
        P2_x = torch.cat((P3_upsampled_x,P2_x),dim=1) #P3_x + P4_upsampled_x
        P2_x = self.P2_2(P2_x)
        
             
        return P2_x
    
class CustomPyramidFeaturesAT2_small2_newLayer_P2(nn.Module):
    def __init__(self, C2_size, C3_size, C4_size, C5_size, layers = [3,4,5,6,7], feature_size=256):
        super(CustomPyramidFeaturesAT2_small2_newLayer_P2, self).__init__()
        
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
        self.Att4 = Attention_block(F_g=feature_size//2, F_l=feature_size//2, F_int=feature_size//2)
        
        self.P4_12_1 = nn.Conv2d(feature_size, feature_size//2, kernel_size=1, stride=1, padding=0)
        
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        

        
        self.P3_1_1 = nn.Conv2d(C3_size, feature_size//2, kernel_size=1, stride=1, padding=0)
        self.P3_1_3 = nn.Conv2d(C3_size, feature_size//2, kernel_size=3, stride=1, padding=1)
        self.P3_1_5 = nn.Conv2d(C3_size, feature_size//2, kernel_size=3, stride=1, padding=2, dilation=2)
        self.Att3 = Attention_block(F_g=feature_size//2, F_l=feature_size//2, F_int=feature_size//2)
        
        self.P3_12_1 = nn.Conv2d(feature_size, feature_size//2, kernel_size=1, stride=1, padding=0)

        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        

        self.P2_1_1 = nn.Conv2d(C2_size, feature_size//2, kernel_size=1, stride=1, padding=0)
        self.P2_1_3 = nn.Conv2d(C2_size, feature_size//2, kernel_size=3, stride=1, padding=1)
        self.P2_1_5 = nn.Conv2d(C2_size, feature_size//2, kernel_size=3, stride=1, padding=2, dilation=2)
        self.Att2 = Attention_block(F_g=feature_size//2, F_l=feature_size//2, F_int=feature_size//2)

        self.P2_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        


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
        P4_upsampled_x = self.P4_12_1(self.P4_upsampled(P4_x))
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1_1(C3) #
        P3_x = P3_x + self.P3_1_3(C3) 
        P3_x = P3_x + self.P3_1_5(C3) 
        P3_x = self.Att3(g=P4_upsampled_x, x=P3_x)
        P3_x = torch.cat((P4_upsampled_x,P3_x),dim=1) #P3_x + P4_upsampled_x
        P3_upsampled_x = self.P3_12_1(self.P3_upsampled(P3_x))
        P3_x = self.P3_2(P3_x)

        P2_x = self.P2_1_1(C2) #
        P2_x = P2_x + self.P2_1_3(C2) 
        P2_x = P2_x + self.P2_1_5(C2) 
        P2_x = self.Att2(g=P3_upsampled_x, x=P2_x)
        P2_x = torch.cat((P3_upsampled_x,P2_x),dim=1) #P3_x + P4_upsampled_x
        P2_x = self.P2_2(P2_x)
        
             
        return P2_x

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class PyramidFeatures(nn.Module):
    def __init__(self, C2_size, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()


        self.P5_1 = nn.Conv2d(C5_size, 1024, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = double_conv(1024, 512)


        self.P4_1 = nn.Conv2d(C4_size, 512, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)


        self.P3_1 = nn.Conv2d(C3_size, 256, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = double_conv(256, 256)
        




    def forward(self, inputs):
        C2, C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_upsampled_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_upsampled_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_x
        P3_upsampled_x = self.P3_upsampled(P3_x)
        P3_x = self.P3_2(P3_upsampled_x)

        return P3_x

class RegressionModel(nn.Module):
    def __init__(self, num_features_in, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv3 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, 2, kernel_size=1, padding=0)

    def forward(self, x):

        out = self.conv3(x)
        out = self.act3(out)

        out = self.output(out)


        return out


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_classes, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes

        self.conv3 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_classes, kernel_size=1, padding=0)


    def forward(self, x):

        out = self.conv3(x)
        out = self.act3(out)

        out = self.output(out)


        return out

    
class ResNet(nn.Module):

    def __init__(self, num_classes, block, layers, inputs = 3):
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
        
        
        

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")
            
        
        
        self.fpn = PyramidFeatures(fpn_sizes[0] // 2, fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])


        self.outc = ClassificationModel(256, self.num_classes)
        self.outr = RegressionModel(256)
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        

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

    def forward(self, img_batch):
            
        batch_size = img_batch.shape[0]

        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        P2 = self.fpn([x1, x2, x3, x4])
        
        outc = self.outc(P2)
        outr = self.outr(P2)

        return outc, outr



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