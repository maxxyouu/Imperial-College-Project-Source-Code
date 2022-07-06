"""
Script to compute the necessary metrics of a model
A.D, A.I, Modified A.I, and segmentation
TODO: Implement AD, AI,
Target layer to be used is accor
"""

from cmath import e
from copy import deepcopy
import torch
import matplotlib.pyplot as plt
import os
from skresnet import skresnext50_32x4d
from layers import *
from RelevanceCamUtils import *
import Constants
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.data.sampler import SequentialSampler
from Helper import denorm
from EvaluatorUtils import *
import logging
from PIL import Image

default_model_name = 'skresnext50_32x4d'
my_parser = argparse.ArgumentParser(description='')
my_parser.add_argument('--model_name',
                        type=str, default=default_model_name,
                        help='model name to be used for model retrival and weight replacement') 
my_parser.add_argument('--model_weights',
                        type=str, default=os.path.join(Constants.SAVED_MODEL_PATH, default_model_name+'_pretrain.pt'),
                        help='Destination for the model weights') 
my_parser.add_argument('--target_layer',
                        type=str, default='layer3',
                        help='cam layer for explanation: target layer to be used can be according to a metrics with --targe_layer = None') 
my_parser.add_argument('--batch_size',
                        type=int, default=3,
                        help='batch size to be used for training / testing') 
my_parser.add_argument('--exp_map_func',
                        type=str, default='hard_threshold_explanation_map',
                        help='match one of the function name') 
my_parser.add_argument('--evaluate_all_layers',
                        type=bool, action=argparse.BooleanOptionalAction, # example: ckpt_epoch_500
                        help='average drop and increase in confidence metrics for each layer')
my_parser.add_argument('--cam',
                        type=str, default='relevance-cam', # example: ckpt_epoch_500
                        help='select a cam')                        
args = my_parser.parse_args()

# Sanity checks for the script arguments
print('Model Name: {}'.format(args.model_name))
print('Model Weight Destination: {}'.format(args.model_weights))
print('Target Layer: {}'.format(args.target_layer))
print('Batch Size: {}'.format(args.batch_size))
args.evaluate_all_layers = False
print('Unbias Layer Selection: {}'.format(args.evaluate_all_layers))
print('Explanation map style: {}'.format(args.exp_map_func))
print('CAM: {}'.format(args.cam))

model = skresnext50_32x4d(pretrained=False).eval()
model.num_classes = 2 #NOTE required to do CLRP and SGLRP
model.fc = Linear(model.fc.in_features, model.num_classes, device=Constants.DEVICE, dtype=Constants.DTYPE)
# load the trained weights
model.load_state_dict(torch.load(args.model_weights, map_location=Constants.DEVICE))
model.to(Constants.DEVICE)
print('Model successfully loaded')

target_layer = args.target_layer
if target_layer == 'layer2':
    target_layer = model.layer2
    size = 29
elif target_layer == 'layer3':
    target_layer = model.layer3
    size = 15
elif target_layer == 'layer4':
    target_layer = model.layer4
    size = 8
else:
    target_layer = model.layer1
    size = 58

value = dict()
def forward_hook(module, input, output):
    value['activations'] = output
def backward_hook(module, input, output):
    value['gradients'] = output[0]

#Feed the data into the model
# data_dir = os.path.join(Constants.STORAGE_PATH, 'mutual_corrects')
data_dir = os.path.join(Constants.STORAGE_PATH, 'picture')

data_transformers = transforms.Compose(
    [
        transforms.ToTensor(), # no need for the centercrop as it is at the cor
        transforms.Normalize(
            [Constants.DATA_MEAN, Constants.DATA_MEAN, Constants.DATA_MEAN], 
            [Constants.DATA_STD,Constants.DATA_STD, Constants.DATA_STD])
    ])
data = datasets.ImageFolder(data_dir, transform=data_transformers)

# for each image, it has a folder that store all the cam heatmaps
sequentialSampler = SequentialSampler(data)
dataloader = DataLoader(data, batch_size=args.batch_size, sampler=sequentialSampler) # TODO: check image 18
image_order_book, img_index = data.imgs, 0
layers = ['layer1', 'layer2', 'layer3', 'layer4']
layer_idx_mapper = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3}


forward_handler = target_layer.register_forward_hook(forward_hook)
backward_handler = target_layer.register_backward_hook(backward_hook)
print('Registered Hooks')

args.exp_map_func = eval(args.exp_map_func)

if args.evaluate_all_layers:
    ad_logger = Average_Drop_logger(np.zeros((1, 4)))
    ic_logger = Increase_Confidence_logger(np.zeros((1, 4)))
else:
    ad_logger = Average_Drop_logger(np.zeros((1,1)))
    ic_logger = Increase_Confidence_logger(np.zeros((1,1)))

for x, y in dataloader:
    # sample_name = image_order_book[img_index][0].split('/')[-1]
    # NOTE: make sure i able index to the correct index
    print('--------- Forward Passing the Original Data ------------')
    x = x.to(device=Constants.DEVICE, dtype=Constants.DTYPE)

    Yci = None
    if args.evaluate_all_layers:
        layer_explanations = [] # each index location store a batch-size of cam explanation map
        for layer in layers:
            cams, Yci = model(x, mode=layer, target_class=[None], internal=False, alpha=2)
            _, Yci = model(x, mode=layer, target_class=[None], internal=False, alpha=2)

            layer_explanations.append(resize_cam(cams[0]))
            Yci = Yci[range(Yci.shape[0]), y].unsqueeze(1) # only care about the score for the true label
        cam = layer_explanations[layer_idx_mapper[args.target_layer]] # retrieve the target layer according to the argument provided for the following code
    else:
        cams, Yci = model(x, mode=args.target_layer, target_class=[None], internal=False, alpha=2)
        cam = resize_cam(cams[0]) # cam map is one dimension in the channel dimention
        Yci = Yci[range(Yci.shape[0]), y].unsqueeze(1)

    img = resize_img(deepcopy(denorm(x)))
    
    # explanation_map = preprocess_image(explanation_map) # transform the data
    print('--------- Forward Passing the Explanation Maps ------------')
    if args.evaluate_all_layers:

        layer_explanation_scores = [] # each index store a batch-size of output scores
        for i, layer in enumerate(layers):

            # get the corresponding explanation map
            cam = layer_explanations[i]
            explanation_map = get_explanation_map(args.exp_map_func, img, cam)

            ## NOTE: FOR DEBUG
            # for j in range(cam.shape[0]):
            #     plt.imshow(cam[j,:].squeeze(0), cmap='seismic')
            #     plt.imshow(np.transpose(img[j,:], (1,2,0)), alpha=.5)
            #     plt.axis('off')

            _, exp_scores = model(explanation_map, mode='output', target_class=[None], internal=False, alpha=2)
            layer_explanation_scores.append(exp_scores[range(Yci.shape[0]), y]) # the corresponding label score (the anchor)
        # [batch_size, layers]
        Oci = torch.stack(layer_explanation_scores, dim=1)

    else:
        explanation_map = get_explanation_map(args.exp_map_func, img, cam)
        _, exp_scores = model(explanation_map, mode=args.target_layer, target_class=[None], internal=False, alpha=2)
        Oci = exp_scores[range(Yci.shape[0]), y].unsqueeze(1)
        # compare the explanation score with the original score

    # collect metrics data
    ad_logger.compute_and_update(Yci.detach().numpy(), Oci.detach().numpy())
    ic_logger.compute_and_update(Yci.detach().numpy(), Oci.detach().numpy())

    forward_handler.remove()
    backward_handler.remove()
    img_index += x.shape[0]

# print the metrics results
print('Average Drop: {}; Average Increase: {}'.format(ad_logger.get_avg(), ic_logger.get_avg()))