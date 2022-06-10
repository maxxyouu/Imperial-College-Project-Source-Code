import torch
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, LayerCAM
from BaselineModel import Pytorch_default_resNet, Pytorch_default_skres, Pytorch_default_vgg
from CLEImageDataset import CLEImageDataset
from Constants import WORK_ENV
import Constants

def mu_std(data_loader):
    count = 0
    mean, var = 0, 0
    for batch_features, _ in data_loader:
        # Rearrange batch to be the shape of [B, C, W * H]
        batch_features = batch_features.view(batch_features.size(0), batch_features.size(1), -1)
        # Update total number of images
        count += batch_features.size(0)
        # Compute mean and std here
        mean += batch_features.mean(2).sum(0) 
        var += batch_features.var(2).sum(0)

    mean /= count
    var /= count
    std = torch.sqrt(var)
    return mean, std


def extract_args():
    my_parser = argparse.ArgumentParser(description='')

    # Add the arguments
    my_parser.add_argument('--model',
                            type=str, default='resnet18',
                            help='model to be used for training / testing')
    my_parser.add_argument('--pretrain',
                            type=bool, default=False,
                            help='whether to use a pretrained model')
    my_parser.add_argument('--batchSize',
                            type=int, default=256,
                            help='batch size to be used for training / testing')             
    my_parser.add_argument('--epochs',
                            type=int, default=100,
                            help='training epochs')   
    my_parser.add_argument('--earlyStoppingPatience',
                            type=int, default=10,
                            help='early stopping patience to terminate the training process')   
    my_parser.add_argument('--learningRate',
                            type=float, default=0.001,
                            help='learning rate for training') 
    my_parser.add_argument('--augNoise',
                            type=bool, default=False,
                            help='add noise during traning')   
    my_parser.add_argument('--train',
                            type=bool, default=True,
                            help='whether execute the script in training or eval mode')   

    # Execute the parse_args() method
    args = my_parser.parse_args()                                              
    return args

def denorm(tensor):
    return tensor.mul(Constants.DATA_STD).add(Constants.DATA_MEAN)

# 'gradcam', 'gradcam++', 'scorecam', 'ablationcam', 'xgradcam', 'eigencam', 'fullgradcam']
def switch_cam(cam, model, target_layers):
    if cam == 'gradcam':
        return GradCAM(model=model, target_layers=target_layers, use_cuda=True if WORK_ENV == 'COLAB' else False)
    elif cam == 'gradcam++':
        return GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True if WORK_ENV == 'COLAB' else False)
    elif cam == 'scorecam':
        return ScoreCAM(model=model, target_layers=target_layers, use_cuda=True if WORK_ENV == 'COLAB' else False)
    elif cam == 'ablationcam':
        return AblationCAM(model=model, target_layers=target_layers, use_cuda=True if WORK_ENV == 'COLAB' else False)
    elif cam == 'xgradcam':
        return XGradCAM(model=model, target_layers=target_layers, use_cuda=True if WORK_ENV == 'COLAB' else False)
    elif cam == 'eigencam':
        return EigenCAM(model=model, target_layers=target_layers, use_cuda=True if WORK_ENV == 'COLAB' else False)
    elif cam == 'fullgradcam':
        return FullGrad(model=model, target_layers=target_layers, use_cuda=True if WORK_ENV == 'COLAB' else False)
    elif cam == 'layercam':
        return LayerCAM(model=model, target_layers=target_layers, use_cuda=True if WORK_ENV == 'COLAB' else False)
    else:
        print('NO SUCH CAM EXISTS')

def main_executation(main, train=True):

    if train:
        print('Training Started')
        main.train()
    else: # eval mode
        print('Testing Started')
        main.check_accuracy(main.loader_test, True, True, True)

def data_transformations():
    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(), 
            transforms.CenterCrop(230), # transforms.CenterCrop((336, 350)), 230 is the number that has the largest square in a circle
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation((0, 270)),
            transforms.RandomAutocontrast(0.25),
            transforms.Normalize(
            [Constants.DATA_MEAN, Constants.DATA_MEAN, Constants.DATA_MEAN], 
            [Constants.DATA_STD,Constants.DATA_STD, Constants.DATA_STD])
        ]
    )

    test_transforms = transforms.Compose([
        transforms.ToTensor(), 
        transforms.CenterCrop(230),
        transforms.Normalize(
            [Constants.DATA_MEAN, Constants.DATA_MEAN, Constants.DATA_MEAN], 
            [Constants.DATA_STD,Constants.DATA_STD, Constants.DATA_STD]
        )
    ])

    return train_transforms, test_transforms

def pytorch_dataset(batch_size, train_transforms, test_transforms):
    if Constants.WORK_ENV == 'COLAB':
        train_datapath = '{}train'.format(Constants.DATA_PARENT_PATH)
        val_datapath = '{}val'.format(Constants.DATA_PARENT_PATH)
        test_datapath = '{}test'.format(Constants.DATA_PARENT_PATH)

        train_annotationPath = '{}train_annotations.csv'.format(Constants.DATA_PARENT_PATH)
        val_annotationPath = '{}val_annotations.csv'.format(Constants.DATA_PARENT_PATH)
        test_annotationPath = '{}test_annotations.csv'.format(Constants.DATA_PARENT_PATH)

    else: # local
        train_datapath = '{}train'.format(Constants.DATA_PARENT_PATH)
        val_datapath = '{}val'.format(Constants.DATA_PARENT_PATH)
        test_datapath = '{}test'.format(Constants.DATA_PARENT_PATH)

        train_annotationPath = '{}train_annotations.csv'.format(Constants.DATA_PARENT_PATH)
        val_annotationPath = '{}val_annotations.csv'.format(Constants.DATA_PARENT_PATH)
        test_annotationPath = '{}test_annotations.csv'.format(Constants.DATA_PARENT_PATH)

    train = CLEImageDataset(train_datapath, annotations_file=train_annotationPath, transform=train_transforms)
    val = CLEImageDataset(val_datapath, annotations_file=val_annotationPath, transform=train_transforms)
    test = CLEImageDataset(test_datapath, annotations_file=test_annotationPath, transform=test_transforms)

    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=True)

    result = {
        'train': [train, train_dataloader],
        'val': [val, val_dataloader],
        'test': [test, test_dataloader]
    }

    return result


def switch_model(model_name, pretrain):
    if 'resnet' in model_name:
        return Pytorch_default_resNet(device=Constants.DEVICE, dtype=Constants.DTYPE, model_name=model_name, pretrain=pretrain)
    if 'vgg' in model_name:
        return Pytorch_default_vgg(device=Constants.DEVICE, dtype=Constants.DTYPE, model_name=model_name, pretrain=pretrain)
    if 'skresnet' in model_name:
        return Pytorch_default_skres(device=Constants.DEVICE, dtype=Constants.DTYPE, model_name=model_name, pretrain=pretrain)