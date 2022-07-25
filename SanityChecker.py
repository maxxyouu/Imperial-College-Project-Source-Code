import argparse
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from skresnet import skresnext50_32x4d
import os
import Constants
from copy import deepcopy
from layers import *

my_parser = argparse.ArgumentParser(description='')

# Add the arguments
default_model_name = 'skresnext50_32x4d'
my_parser.add_argument('--model',
                        type=str, default=default_model_name,
                        help='model to be used for training / testing')
my_parser.add_argument('--batch_size',
                        type=int, default=2,
                        help='batch size to be used for training / testing')
my_parser.add_argument('--cam',
                        type=str, default='relevanceCam',
                        help='cam string')
my_parser.add_argument('--data_location',
                        type=str, default=Constants.ANNOTATED_IMG_PATH, #os.path.join(Constants.STORAGE_PATH, 'mutual_corrects'), # for segmentation: Constants.ANNOTATED_IMG_PATH
                        help='data directory')  
my_parser.add_argument('--model_weights',
                        type=str, default=os.path.join(Constants.SAVED_MODEL_PATH, default_model_name+'_pretrain.pt'),
                        help='Destination for the model weights')
my_parser.add_argument('--independentRandomFolder',
                        type=str, default='',
                        help='Destination for final results') 
my_parser.add_argument('--cascadingRandomFolder',
                        type=str, default='',
                        help='Destination for final results') 
args = my_parser.parse_args()


# Sanity check of the arguments
print('model name {}'.format(args.model))
print('batch size: {}'.format(args.batch_size))
print('Data Location {}'.format(args.data_location))


# transformation needed for the input images
eval_transforms = transforms.Compose([
    transforms.ToTensor(), 
    transforms.CenterCrop(230),
    transforms.Normalize(
        [Constants.DATA_MEAN, Constants.DATA_MEAN, Constants.DATA_MEAN], 
        [Constants.DATA_STD,Constants.DATA_STD, Constants.DATA_STD]
    )
])

# prepare dataset
data = datasets.ImageFolder(args.data_location, transform=eval_transforms)
sequentialSampler = SequentialSampler(data)
dataloader = DataLoader(data, batch_size=args.batch_size, sampler=sequentialSampler) # TODO: check image 18
image_order_book, img_index = data.imgs, 0

# prepare the model
model = skresnext50_32x4d(pretrained=True)
model.num_classes = 2
model.fc = Linear(model.fc.in_features, model.num_classes, device=Constants.DEVICE, dtype=Constants.DTYPE)
model.load_state_dict(torch.load(args.model_weights, map_location=Constants.DEVICE))
model.to(Constants.DEVICE)
model.eval() # after loading the model, put the model into evaluation mode
print('Model successfully loaded')

# hooks for the relevance
value = dict()
def forward_hook(module, input, output):
    value['activations'] = output
def backward_hook(module, input, output):
    value['gradients'] = output[0]
fhs, bhs = [], []
for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
    fh = getattr(model, layer).register_forward_hook(forward_hook)
    bh = getattr(model, layer).register_full_backward_hook(backward_hook)
    fhs.append(fh)
    bhs.append(bh)
print('Register hooks successful')

# create folder for independent weight randomization and cascading weight randomization
if not os.path.exists(args.independentRandomFolder):
    os.makedirs(args.independentRandomFolder)
if not os.path.exists(args.cascadingRandomFolder):
    os.makedirs(args.cascadingRandomFolder)

print('Performing Cascading Randomization Test')
# cascading randomization test


print('Performing Independent Randomization Test')
# independent randomization test

#NOTE: each randomized weight with a copy of the network




