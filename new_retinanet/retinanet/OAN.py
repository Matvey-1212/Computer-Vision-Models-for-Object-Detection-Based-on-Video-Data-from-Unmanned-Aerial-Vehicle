import torch
import torch.nn as  nn
import torch.nn.functional as F
from retinanet import losses
import math


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        
        x = self.relu(self.batch_norm2(self.conv2(x)))
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        x+=identity
        x=self.relu(x)
        
        return x

class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()
       

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
      identity = x.clone()

      x = self.relu(self.batch_norm2(self.conv1(x)))
      x = self.batch_norm2(self.conv2(x))

      if self.i_downsample is not None:
          identity = self.i_downsample(identity)
      print(x.shape)
      print(identity.shape)
      x += identity
      x = self.relu(x)
      return x


        
        
class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64
        #Resnet50
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        
        # self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # self.fc = nn.Linear(512*ResBlock.expansion, num_classes)
        
        #OAN
        
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
        
        # self.oan_batch_norm1 = nn.BatchNorm2d(1)
        
        self.loss = losses.OANFocalLoss(alpha = 0.25, gamma = 2)
        
        self.oan_batch_norm3 = nn.BatchNorm2d(1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        
        
    def forward(self, inputs):
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs
            
        shape = img_batch.shape

        x = self.relu(self.batch_norm1(self.conv1(img_batch)))
        
        # allocated_memory = torch.cuda.memory_allocated(device)
        # cached_memory = torch.cuda.memory_reserved(device)
        # print(f"2 Выделенная память: {allocated_memory / (1024 ** 3)} GB")
        # print(f"2 Закэшированная память: {cached_memory / (1024 ** 3)} GB")
        x = self.max_pool(x)

        # allocated_memory = torch.cuda.memory_allocated(device)
        # cached_memory = torch.cuda.memory_reserved(device)
        # print(f"3 Выделенная память: {allocated_memory / (1024 ** 3)} GB")
        # print(f"3 Закэшированная память: {cached_memory / (1024 ** 3)} GB")
        x = self.layer1(x)

        # allocated_memory = torch.cuda.memory_allocated(device)
        # cached_memory = torch.cuda.memory_reserved(device)
        # print(f"4 Выделенная память: {allocated_memory / (1024 ** 3)} GB")
        # print(f"4 Закэшированная память: {cached_memory / (1024 ** 3)} GB")
        x = self.layer2(x)

        # allocated_memory = torch.cuda.memory_allocated(device)
        # cached_memory = torch.cuda.memory_reserved(device)
        # print(f"5 Выделенная память: {allocated_memory / (1024 ** 3)} GB")
        # print(f"5 Закэшированная память: {cached_memory / (1024 ** 3)} GB")
        x = self.layer3(x)

        # allocated_memory = torch.cuda.memory_allocated(device)
        # cached_memory = torch.cuda.memory_reserved(device)
        # print(f"6 Выделенная память: {allocated_memory / (1024 ** 3)} GB")
        # print(f"6 Закэшированная память: {cached_memory / (1024 ** 3)} GB")
        x = self.layer4(x)

        # allocated_memory = torch.cuda.memory_allocated(device)
        # cached_memory = torch.cuda.memory_reserved(device)
        # print(f"7 Выделенная память: {allocated_memory / (1024 ** 3)} GB")
        # print(f"7 Закэшированная память: {cached_memory / (1024 ** 3)} GB")
        
        x = self.oan_layer1(x)
        # allocated_memory = torch.cuda.memory_allocated(device)
        # cached_memory = torch.cuda.memory_reserved(device)
        # print(f"8 Выделенная память: {allocated_memory / (1024 ** 3)} GB")
        # print(f"8 Закэшированная память: {cached_memory / (1024 ** 3)} GB")

        x = self.oan_layer2(x)
        # allocated_memory = torch.cuda.memory_allocated(device)
        # cached_memory = torch.cuda.memory_reserved(device)
        # print(f"9 Выделенная память: {allocated_memory / (1024 ** 3)} GB")
        # print(f"9 Закэшированная память: {cached_memory / (1024 ** 3)} GB")

        x = self.oan_layer3(x)
        # allocated_memory = torch.cuda.memory_allocated(device)
        # cached_memory = torch.cuda.memory_reserved(device)
        # print(f"10 Выделенная память: {allocated_memory / (1024 ** 3)} GB")
        # print(f"10 Закэшированная память: {cached_memory / (1024 ** 3)} GB")

        x = self.oan_batch_norm3(x)
        # allocated_memory = torch.cuda.memory_allocated(device)
        # cached_memory = torch.cuda.memory_reserved(device)
        # print(f"11 Выделенная память: {allocated_memory / (1024 ** 3)} GB")
        # print(f"11 Закэшированная память: {cached_memory / (1024 ** 3)} GB")
        # x = self.oan_batch_norm1(x)
        
        x = F.softmax(x, dim=-1)
        # allocated_memory = torch.cuda.memory_allocated(device)
        # cached_memory = torch.cuda.memory_reserved(device)
        # print(f"12 Выделенная память: {allocated_memory / (1024 ** 3)} GB")
        # print(f"12 Закэшированная память: {cached_memory / (1024 ** 3)} GB")

        if self.training:
            return self.loss(x, annotations, (shape[-2], shape[-1]))
        return x
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)

def OAN(channels=3):
    return ResNet(Bottleneck, [3,4,6,3], channels)
        
# def ResNet50(num_classes, channels=3):
#     return ResNet(Bottleneck, [3,4,6,3], num_classes, channels)
    
# def ResNet101(num_classes, channels=3):
#     return ResNet(Bottleneck, [3,4,23,3], num_classes, channels)

# def ResNet152(num_classes, channels=3):
#     return ResNet(Bottleneck, [3,8,36,3], num_classes, channels)

