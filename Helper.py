from xmlrpc.client import boolean
import torch
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, LayerCAM
# from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, LayerCAM
from BaselineModel import Pytorch_default_resNet, Pytorch_default_resnext, Pytorch_default_skres, Pytorch_default_skresnext, Pytorch_default_vgg
from CLEImageDataset import CLEImageDataset
from Constants import WORK_ENV
import Constants
import os
import shutil


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
                            type=str, default='skresnext50_32x4d',
                            help='model to be used for training / testing')
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
    my_parser.add_argument('--pretrain',
                            type=bool, action=argparse.BooleanOptionalAction,
                            help='whether to use a pretrained model')
    my_parser.add_argument('--augNoise',
                            type=bool, action=argparse.BooleanOptionalAction,
                            help='add noise during traning')   
    my_parser.add_argument('--train',
                            type=bool, action=argparse.BooleanOptionalAction,
                            help='whether execute the script in training or eval mode')   
    my_parser.add_argument('--chkPointName',
                            type=str, default='last.pth', # example: ckpt_epoch_500
                            help='the check point name')  
    my_parser.add_argument('--simClr',
                            type=bool, action=argparse.BooleanOptionalAction, # example: ckpt_epoch_500
                            help='for simclr task') 
    # Execute the parse_args() method
    args = my_parser.parse_args()                                              
    return args

def extract_attention_cam_args():
    my_parser = argparse.ArgumentParser(description='')
    my_parser.add_argument('--model',
                            type=str, default='skresnext50_32x4d',
                            help='model to be used for training / testing') 
    my_parser.add_argument('--noiseSmooth',
                            type=bool, action=argparse.BooleanOptionalAction,
                            help='use noise for smoothing or not') 
    my_parser.add_argument('--iterations',
                            type=int, default=50,
                            help='iteration for smoothing') 
    my_parser.add_argument('--std',
                            type=float, default=1.,
                            help='noise level for smoothing') 
    my_parser.add_argument('--cam',
                            type=str, default='xgradcam',
                            help='cam name for explanation') 
    my_parser.add_argument('--layers',
                            type=int, default=4,
                            help='cam name for explanation') 
    my_parser.add_argument('--batchSize',
                            type=int, default=256,
                            help='batch size to be used for training / testing') 
    my_parser.add_argument('--positiveTarget',
                            type=bool, action=argparse.BooleanOptionalAction,
                            help='generate cam from positive target')  
 
    # 'scorecam', 'ablationcam', 'xgradcam', 'eigencam',
    
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

    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers= 2 if Constants.WORK_ENV == 'COLAB' else 0)
    val_dataloader = DataLoader(val, batch_size=batch_size, shuffle=True, num_workers= 2 if Constants.WORK_ENV == 'COLAB' else 0)
    test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=True, num_workers= 2 if Constants.WORK_ENV == 'COLAB' else 0)

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
    if 'skresnext' in model_name:
        return Pytorch_default_skresnext(device=Constants.DEVICE, dtype=Constants.DTYPE, model_name=model_name, pretrain=pretrain)
    if 'resnext' in model_name:
        return Pytorch_default_resnext(device=Constants.DEVICE, dtype=Constants.DTYPE, model_name=model_name, pretrain=pretrain)
    print('NO MATCHED MODEL')

def get_trained_model(model_name):
    # get actual model name without the _pretrain suffix
    _PRETRAIN = '_pretrain'
    model_name_cleaned = model_name
    if _PRETRAIN in model_name:
        end = model_name.index(_PRETRAIN)
        model_name_cleaned = model_name[:end]
    
    model_wrapper = switch_model(model_name_cleaned, False)
    weight_pickle = os.path.join(Constants.SAVED_MODEL_PATH ,'{}.pt'.format(model_name))
    model_wrapper.load_learned_weights(weight_pickle)
    
    return model_wrapper
    
def find_mutual_correct_images(dest_root):
    # root 
    root_path = os.path.join(Constants.STORAGE_PATH, 'correct_preds')

    # correct class prediction for each model
    meingioma_dict = {}
    gbm_dict = {}
    files_path_dict = {}
    for root, dirs, files in os.walk(root_path):
        # reached the leaf nodes
        if len(files) <= 0 or len(dirs) > 0:
            continue

        # use this to retrieve the file after intersection later on
        for file in files:
            files_path_dict[file] = os.path.join(root)

        if root[-1] == '1':
            meingioma_dict[root] = set(files)
        elif root[-1] == '0':
            gbm_dict[root] = set(files)

    # find intersection of correct predictions
    meingiomas = meingioma_dict.values()
    mutual_correct_meingioma = set.intersection(*meingiomas)

    gbms = gbm_dict.values()
    mutual_correct_gbm = set.intersection(*gbms)

    # create subdirectory if necessary
    dest_0 = os.path.join(dest_root, '0')
    dest_1 = os.path.join(dest_root, '1')
    if not os.path.exists(dest_0):
        os.makedirs(dest_0)
    if not os.path.exists(dest_1):
        os.makedirs(dest_1)


    # move file to the desired destination
    union_sets = mutual_correct_meingioma | mutual_correct_gbm
    for file in union_sets:
        root = files_path_dict[file]
        source_path = os.path.join(root, file)
        
        # default is class 0 file and change if necessary
        dest_path = os.path.join(dest_0, file)
        if root[-1] == '1' and os.path.isfile(source_path):
            dest_path = os.path.join(dest_1, file)
        # copy files to destination
        shutil.copy(source_path, dest_path)  

    return
