import torch
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import collater, ToTorch, Augmenter, Normalizer, UnNormalizer
from retinanet.datasetLLAD import LLAD
import torchvision.transforms as T
from retinanet import csv_eval
import pandas as pd



print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):

    
    iou_threshold = 0.5
    model_path = '/home/maantonov_1/VKR/weights/retinanet/retinanet_gwd.pt'
    save_path = '/home/maantonov_1/VKR/actual_scripts/smartcrop_retinanet'

    DIR_TRAIN = '/home/maantonov_1/VKR/data/small_train/full_data/images'
    valid_df = pd.read_csv('/home/maantonov_1/VKR/data/small_train/test_small.csv')

    dataset_val = LLAD(valid_df, DIR_TRAIN, mode = "valid", transforms = T.Compose([ ToTorch()]))
    # Create the model
    #retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)
    retinanet=torch.load(model_path)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        #retinanet.load_state_dict(torch.load(parser.model_path))
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet.load_state_dict(torch.load(model_path))
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = False
    retinanet.eval()
    retinanet.module.freeze_bn()

    print(csv_eval.evaluate(dataset_val, retinanet,iou_threshold=float(iou_threshold)), save_path = save_path)



if __name__ == '__main__':
    main()
