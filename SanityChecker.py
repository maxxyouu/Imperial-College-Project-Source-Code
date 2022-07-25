import argparse
from typing import OrderedDict
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from skresnet import skresnext50_32x4d
import os
import Constants
from copy import deepcopy
from layers import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
import logging

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
my_parser.add_argument('--alpha',
                        type=float, default=1,
                        help='alpha value for propagation')
my_parser.add_argument('--data_location',
                        type=str, default=Constants.ANNOTATED_IMG_PATH, #os.path.join(Constants.STORAGE_PATH, 'mutual_corrects'), # for segmentation: Constants.ANNOTATED_IMG_PATH
                        help='data directory')  
my_parser.add_argument('--model_weights',
                        type=str, default=os.path.join(Constants.SAVED_MODEL_PATH, default_model_name+'_pretrain.pt'),
                        help='Destination for the model weights')
my_parser.add_argument('--independentRandomFolder',
                        type=str, default=os.path.join(Constants.STORAGE_PATH, 'independentRandomFolder'),
                        help='Destination for final results') 
my_parser.add_argument('--cascadingRandomFolder',
                        type=str, default=os.path.join(Constants.STORAGE_PATH, 'cascadingRandomFolder'),
                        help='Destination for final results') 
my_parser.add_argument('--sanityCheckMode',
                        type=str, default='cascade',
                        help='cascade or independent') 
args = my_parser.parse_args()


# Sanity check of the arguments
print('model name {}'.format(args.model))
print('batch size: {}'.format(args.batch_size))
print('Data Location {}'.format(args.data_location))
CHOSEN_ALPHA = args.alpha
print('alpha {}'.format(CHOSEN_ALPHA))

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

# create folder for independent weight randomization and cascading weight randomization
if not os.path.exists(args.independentRandomFolder):
    os.makedirs(args.independentRandomFolder)
if not os.path.exists(args.cascadingRandomFolder):
    os.makedirs(args.cascadingRandomFolder)

# helper function for the following
def randomize_layer_weights(trained_weights, layer_name='fc.'):
    """
    helper function to randomized weight for a particular layer(stage)
    Args:
        trained_weights (from torch.load): ordered dictionary of trained weights
        layer_name (str): use this to filter out which weight should be randomized
            'logit', 'layer4', 'layer3', 'layer2', 'layer1'
    """
    trained_weights_cpy = deepcopy(trained_weights)
    partial_random_weights = []
    for key, tensor_weights in trained_weights_cpy.items():
        # only consider tensor_weights.dtype == Constants.DTYPE == float32 instead of long
        if layer_name == key[:len(layer_name)] and tensor_weights.dtype == Constants.DTYPE:
            randomized_weights = torch.randn_like(tensor_weights, dtype=tensor_weights.dtype, device=tensor_weights.device, requires_grad=tensor_weights.requires_grad)
            partial_random_weights.append((key, randomized_weights))
        else:
            # copy the weights directly
            partial_random_weights.append((key, tensor_weights))

    return OrderedDict(partial_random_weights)
    

print('Performing Cascading Randomization Test')
# cascading randomization test in a unit of layer(stage)
layer_names = ['fc.', 'layer4.', 'layer3.', 'layer2.'] # fc is the logit, visualize the cam of layer 1
trained_weights = torch.load(args.model_weights, map_location=Constants.DEVICE)
cascade_randomized_weights = []
for layer_name in layer_names:
    # cascade randomization
    trained_weights = randomize_layer_weights(trained_weights, layer_name=layer_name)
    cascade_randomized_weights.append(trained_weights)

print('Performing Independent Randomization Test')
trained_weights = torch.load(args.model_weights, map_location=Constants.DEVICE)
independent_randomized_weights = []
# independent randomization test
for layer_name in layer_names:
    _copy = randomize_layer_weights(trained_weights, layer_name=layer_name)
    independent_randomized_weights.append(_copy)


# prepare the model
model = skresnext50_32x4d(pretrained=True)
model.num_classes = 2
model.fc = Linear(model.fc.in_features, model.num_classes, device=Constants.DEVICE, dtype=Constants.DTYPE)

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

destination = args.cascadingRandomFolder

# custom_randomized_weights = cascade_randomized_weights[0]
# model.load_state_dict(custom_randomized_weights)
# model.to(Constants.DEVICE)
# model.eval() # after loading the model, put the model into evaluation mode
# print('Model successfully loaded')

def generate_cam_from_randomized_weights(x, y, model, randomized_weights, layer_names, dest_folder):
    global img_index

    # assume that layer_names and randomized_weights are of the same order
    for randomized_layer_name, custom_randomized_weights in zip(layer_names, randomized_weights):
        model.load_state_dict(custom_randomized_weights)
        model.to(Constants.DEVICE)
        model.eval() # after loading the model, put the model into evaluation mode
        r_cams, _ = model(x, 'layer1', target_class=y, internal=False, alpha=CHOSEN_ALPHA)

        for i in range(x.shape[0]):

            # create a desintation folder using the sample name
            sample_name = image_order_book[img_index][0].split('/')[-1] # get the image name from the dataset
            dest = os.path.join(dest_folder, '0' if y[i].item() == 0 else '1', sample_name)
            
            img = x[i, :]
            img = np.transpose(img, (1,2,0))
            if Constants.WORK_ENV == 'COLAB':
                img = img.cpu().detach().numpy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            plt.ioff()

            logger = logging.getLogger()
            old_level = logger.level
            logger.setLevel(100)

            if Constants.WORK_ENV == 'COLAB':
                r_cam = r_cams[i, :].reshape(58, 58).cpu().detach().numpy()
            else:
                r_cam = r_cams[i, :].reshape(58, 58).detach().numpy()

            r_cam = cv2.resize(r_cam, (230, 230))
            mask = plt.imshow(r_cam, cmap='seismic')
            overlayed_image = plt.imshow(img, alpha=.5)
            plt.axis('off')
            plt.savefig(os.path.join(dest, randomized_layer_name+'.png'))

            img_index += 1

for i, (x, y) in enumerate(dataloader):
    # NOTE: make sure i able index to the correct index
    print('--------- Forward Passing the Original Data ------------')
    x = x.to(device=Constants.DEVICE, dtype=Constants.DTYPE)
    
    # for the cascading case
    if args.sanityCheckMode == 'cascade':
        generate_cam_from_randomized_weights(x, y, model, cascade_randomized_weights, layer_names, dest_folder=args.cascadingRandomFolder)
    elif args.sanityCheckMode == 'independent':
        generate_cam_from_randomized_weights(x, y, model, independent_randomized_weights, layer_names, dest_folder=args.independentRandomFolder)