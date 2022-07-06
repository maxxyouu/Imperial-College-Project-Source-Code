from email.policy import default
from re import X
from torchvision import transforms, datasets
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.utils.data import DataLoader
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np
import os
import torchvision
from torch.utils.data.sampler import SequentialSampler
import torch
from PIL import Image
import matplotlib.pyplot as plt
import logging
import argparse
from Helper import *
from EvaluatorUtils import *

# local imports
import Constants
from Helper import denorm, switch_cam, extract_attention_cam_args, get_trained_model, find_mutual_correct_images
        
def add_noise(x, noise_level):
    noise = np.random.normal(0.0, scale=noise_level)
    noise = torch.tensor(noise, device=Constants.DEVICE, dtype=Constants.DTYPE)
    return x + noise

def define_model_dir_path(args):
    model_dir_name = args.model + '_noiseSmooth' if args.noiseSmooth else args.model
    if args.noiseSmooth:
        model_dir_name += '_noise{}_iters{}'.format(args.std, args.iterations)
    return model_dir_name

# TODO: the following only valid for resnet 50 and resnext 50
def target_layers(model, layer_nums):
    return [getattr(model, 'layer'+'{}'.format(layer_nums))[-1]]

def get_targets(positive):
    cam_targets = None
    if positive is not None and not positive: 
        # generate non-positive targets
        cam_targets = [ClassifierOutputTarget(1 if target == 0 else 0) for target in y.tolist()]
    # else:
    #     cam_targets = [ClassifierOutputTarget(target) for target in y.tolist()]
    return cam_targets

def generate_cam_overlay(x, args, cam, cam_targets):
    input_x = x
    if args.noiseSmooth:
        grayscale_cam = torch.zeros((x.shape[0], x.shape[-1], x.shape[-1]),dtype=Constants.DTYPE)
        for t in range(args.iterations):
            # print('CAM Smoothing Iteration: {}'.format(t))
            input_x = add_noise(x, args.std)
            grayscale_cam += cam(input_tensor=input_x, targets=cam_targets)
        grayscale_cam /= args.iterations
    else:
        grayscale_cam = cam(input_tensor=input_x, targets=cam_targets)
    return grayscale_cam

def threshold(x):
    mean_ = x.mean()
    std_ = x.std()
    thresh = mean_ +std_
    x = (x>thresh)
    return x

default_model_name = 'skresnext50_32x4d'
my_parser = argparse.ArgumentParser(description='')
my_parser.add_argument('--model',
                        type=str, default=default_model_name,
                        help='model to be used for training / testing') 
my_parser.add_argument('--model_weights',
                        type=str, default=os.path.join(Constants.SAVED_MODEL_PATH, default_model_name+'_pretrain.pt'),
                        help='Destination for the model weights') 
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
                        type=int, default=3,
                        help='batch size to be used for training / testing')  
my_parser.add_argument('--run_mode',
                        type=str, default='metrics',
                        help='Metrics mode either "explanation" or "metrics"') 
my_parser.add_argument('--exp_map_func',
                        type=str, default='hard_threshold_explanation_map',
                        help='match one of the function name') 

# 'scorecam', 'ablationcam', 'xgradcam', 'eigencam',
args = my_parser.parse_args()

# print statement to verify the boolean arguments
print('Noise Smooth Arg: {}'.format(args.noiseSmooth))

# model_wrapper = get_trained_model(args.model)
model_wrapper = switch_model(args.model, False)
model_wrapper.load_learned_weights(args.model_weights)
print('successfully load the model')
model_target_layer = target_layers(model_wrapper.model, args.layers) # for script argument input
# model_target_layer = [*model_wrapper.model.layer1, *model_wrapper.model.layer2, *model_wrapper.model.layer3, *model_wrapper.model.layer4]

model_dir_name = define_model_dir_path(args)
# data_dir = os.path.join(Constants.STORAGE_PATH, 'mutual_corrects')
data_dir = os.path.join(Constants.STORAGE_PATH, 'picture')

data = datasets.ImageFolder(data_dir, transform=transforms.Compose(
    [
        transforms.ToTensor(), # no need for the centercrop as it is at the cor
        transforms.Normalize(
            [Constants.DATA_MEAN, Constants.DATA_MEAN, Constants.DATA_MEAN], 
            [Constants.DATA_STD,Constants.DATA_STD, Constants.DATA_STD])
    ]
))
# for each image, it has a folder that store all the cam heatmaps
sequentialSampler = SequentialSampler(data)
dataloader = DataLoader(data, batch_size=args.batchSize, sampler=sequentialSampler) # TODO: check image 18

image_order_book, img_index = data.imgs, 0
cam = switch_cam(args.cam, model_wrapper.model, model_target_layer)
args.exp_map_func = eval(args.exp_map_func) # NOTE

ad_logger = Average_Drop_logger(np.zeros((1,1)))
ic_logger = Increase_Confidence_logger(np.zeros((1,1)))

for x, y in dataloader:
    x = x.to(device=Constants.DEVICE, dtype=Constants.DTYPE)
    
    # NOTE: make sure i able index to the correct index
    print('--------- Forward Passing {}'.format(args.cam))
    # generate batch-wise cam
    cam_targets = None
    grayscale_cam = generate_cam_overlay(x, args, cam, cam_targets)
    
    if args.run_mode == 'metrics':
        print('Forward Passing the original images')
        Yci = model_wrapper.model(x)
        Yci = Yci[range(Yci.shape[0]), y].unsqueeze(1) # get the score respects to the corresponding label
        
        print('Forward Passing the explanation images')
        img = denorm(x).detach().numpy()
        grayscale_cam = np.expand_dims(grayscale_cam, 1)
        explanation_map = get_explanation_map(args.exp_map_func, img, grayscale_cam)
        exp_scores = model_wrapper.model(explanation_map)
        Oci = exp_scores[range(Yci.shape[0]), y].unsqueeze(1)

        # collect metrics data
        ad_logger.compute_and_update(Yci.detach().numpy(), Oci.detach().numpy())
        ic_logger.compute_and_update(Yci.detach().numpy(), Oci.detach().numpy())
        print('Progress: A.D: {}, I.C: {}'.format(ad_logger.current_metrics, ic_logger.current_metrics))
    else:
        # denormalize the image NOTE: must be placed after forward passing
        x = denorm(x)
        
        print('--------- Generating {} Heatmap'.format(args.cam))
        # for each image in a batch
        for i in range(x.shape[0]):
            sample_name = image_order_book[img_index][0].split('/')[-1] # get the image name from the dataset

            # each image is a directory that contains all the experiment results
            dest = os.path.join(Constants.STORAGE_PATH, 'heatmaps', model_dir_name, '0' if y[i].item() == 0 else '1', sample_name)

            # save the original image in parallel
            if not os.path.exists(dest):
                os.makedirs(dest)
                # save the original image
                torchvision.utils.save_image(x[i, :], os.path.join(dest, 'original.jpg'))

            # swap the axis so that the show_cam_on_image works
            img = x[i, :].cpu().detach().numpy()
            img = np.transpose(img, (1,2,0))

            # save the overlayed-attention map with the cam name as a tag
            attention_map = show_cam_on_image(img, grayscale_cam[i, :], use_rgb=True)
            cam_name = '{}-layers{}'.format(args.cam, args.layers)
            
            plt.ioff()

            logger = logging.getLogger()
            old_level = logger.level
            logger.setLevel(100)

            segmentation = plt.imshow(grayscale_cam[i, :], cmap='seismic')
            overlayed_image = plt.imshow(img, alpha=.5)
            plt.axis('off')
            plt.savefig(os.path.join(dest, cam_name+'_seismic.png'))

            segmented_image = img*threshold(grayscale_cam[i, :])[...,np.newaxis]
            segmented_image = np.where(segmented_image == 0, 100, segmented_image)
            segmented_image = plt.imshow(segmented_image)
            plt.axis('off')
            plt.savefig(os.path.join(dest, cam_name+'_segments.png'))
            plt.close()
            
            logger.setLevel(old_level)

            masked_img = Image.fromarray(attention_map, 'RGB')
            masked_img.save(os.path.join(dest, cam_name+'_rgb.jpg'))


            # update the sequential index for next iterations
            img_index += 1

if args.run_mode == 'metrics':
    print('Average Drop: {}; Average Increase: {}'.format(ad_logger.get_avg(), ic_logger.get_avg()))