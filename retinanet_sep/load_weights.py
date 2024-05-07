
import torch.utils.model_zoo as model_zoo

weights = model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth', model_dir='.')