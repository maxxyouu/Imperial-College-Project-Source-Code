from copy import deepcopy
from torchvision import transforms, datasets
from torch.nn.functional import softmax
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
from skresnet import skresnext50_32x4d # this is the version that has dropout in this branch
from layers import *

# local imports
import Constants
from Helper import denorm, switch_cam, extract_attention_cam_args, get_trained_model, find_mutual_correct_images
        
torch.manual_seed(100)
def add_noise(x, noise_level):
    noise = np.random.normal(0.0, scale=noise_level)
    noise = torch.tensor(noise, device=Constants.DEVICE, dtype=Constants.DTYPE)
    return x + noise

def define_model_dir_path(args):
    model_dir_name = args.model + '_noiseSmooth' if args.noiseSmooth else args.model
    if args.noiseSmooth:
        model_dir_name += '_noise{}_iters{}'.format(args.std, args.iterations)
    return model_dir_name

# TODO: the following only valid for resnet and its variant
def target_layers(model, layer_nums):
    results = []
    layers = layer_nums.split(',')
    for layer_num in layers:# in layer number order
        results.append(getattr(model, 'layer'+'{}'.format(layer_num))[-1])
    return results

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
    # NOTE: threshold per images
    mean_ = np.mean(x, axis=(1, 2), keepdims=True)
    std_ = np.std(x, axis=(1,2), keepdims=True)
    thresh = mean_ +std_
    x = (x>thresh)
    return x

default_model_name = 'skresnext50_32x4d'
my_parser = argparse.ArgumentParser(description='')
my_parser.add_argument('--model',
                        type=str, default=default_model_name,
                        help='model to be used for training / testing') 
my_parser.add_argument('--model_weights',
                        type=str, default='',
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
                        type=str, default='gradcam',
                        help='cam name for explanation') 
my_parser.add_argument('--layers',
                        type=str, default='3,4',
                        help='cam name for explanation') 
my_parser.add_argument('--batchSize',
                        type=int, default=2,
                        help='batch size to be used for training / testing')  
my_parser.add_argument('--run_mode',
                        type=str, default='metrics',
                        help='Metrics mode either "explanation" or "metrics"') 
my_parser.add_argument('--exp_map_func',
                        type=str, default='hard_threshold_explanation_map',
                        help='match one of the function name') 
my_parser.add_argument('--data_location',
                        type=str, default=Constants.ANNOTATED_IMG_PATH, #os.path.join(Constants.STORAGE_PATH, 'mutual_corrects'), # example: ckpt_epoch_500
                        help='data directory') 
my_parser.add_argument('--eval_segmentation',
                        type=bool, action=argparse.BooleanOptionalAction,
                        help='true indicate evaluate the segmentation performance of the cam method')
my_parser.add_argument('--headWidth',
                        type=int, default=1,
                        help='true indicate evaluate the segmentation performance of the cam method')
my_parser.add_argument('--annotation_path',
                        type=str, default=Constants.ANNOTATION_PATH,
                        help='path for the imge annotation')
my_parser.add_argument('--eval_model_uncertainty',
                        type=bool, action=argparse.BooleanOptionalAction,
                        help='evaluate model uncertainty')
my_parser.add_argument('--ensemble_N',
                        type=int, default=2,
                        help='number of times to evaluate using the layer dropout, only be used with eval_model_uncertainty holds true')

# 'scorecam', 'ablationcam', 'xgradcam', 'eigencam',
args = my_parser.parse_args()

# print statement to verify the boolean arguments
print('Noise Smooth Arg: {}'.format(args.noiseSmooth))
print('Model Name: {}'.format(args.model))
print('Target Layer: {}'.format(args.layers))
print('Batch Size: {}'.format(args.batchSize))
print('Explanation map style: {}'.format(args.exp_map_func))
print('CAM: {}'.format(args.cam))
print('Data Location {}'.format(args.data_location))
print('Head Width: {}'.format(args.headWidth))
if Constants.WORK_ENV == 'LOCAL': # NOTE: FOR DEBUG PURPOSE
    args.eval_model_uncertainty = True
if args.eval_model_uncertainty is None:
    args.eval_model_uncertainty = False
elif args.eval_model_uncertainty and args.model_weights == '' :
    args.eval_segmentation = False
    #make sure in the correct data source location
    assert(args.data_location == Constants.ANNOTATED_IMG_PATH)
    assert(args.annotation_path == Constants.ANNOTATION_PATH)
    annotation_file_list = os.listdir(args.annotation_path)
    args.model_weights = os.path.join(Constants.SAVED_MODEL_PATH, args.model +'_headWidth1_withLayerDropout_pretrain.pt')
print('Evaluate Model Uncertainty: {}'.format(args.eval_model_uncertainty))
if args.model_weights == '': # default model weight destination
    args.model_weights = os.path.join(Constants.SAVED_MODEL_PATH, args.model +'_pretrain.pt')
print('Model Weight Destination: {}'.format(args.model_weights))
data_dir = args.data_location


if Constants.WORK_ENV == 'LOCAL': # NOTE: FOR DEBUG PURPOSE
    args.eval_segmentation = True 
if args.eval_segmentation is None:
    args.eval_segmentation = False
else:
    #make sure in the correct data source location
    assert(args.data_location == Constants.ANNOTATED_IMG_PATH)
    assert(args.annotation_path == Constants.ANNOTATION_PATH)
    annotation_file_list = os.listdir(args.annotation_path)
print('Evaluate Segmentation {}'.format(args.eval_segmentation))

# model_wrapper = get_trained_model(args.model)
if args.model == 'skresnext50_32x4d':
    model_wrapper = switch_model(args.model, False, headWidth=args.headWidth)
    model = skresnext50_32x4d(pretrained=False)
    model.num_classes = 2
    model.fc = Linear(model.fc.in_features, model.num_classes, device=Constants.DEVICE, dtype=Constants.DTYPE)
    model_wrapper.model = deepcopy(model)
else:
    model_wrapper = switch_model(args.model, False, headWidth=args.headWidth)

model_wrapper.load_learned_weights(args.model_weights)
if not args.eval_model_uncertainty:
    model_wrapper.model.eval() # put the model into evaluation mode for the dropout layer
    print('In evaluationg mode')
else:
    model_wrapper.model.train()
    print('In training mode')
print('successfully load the model')
model_target_layer = target_layers(model_wrapper.model, args.layers) # for script argument input
# model_target_layer = [*model_wrapper.model.layer1, *model_wrapper.model.layer2, *model_wrapper.model.layer3, *model_wrapper.model.layer4]

model_dir_name = define_model_dir_path(args)

data = datasets.ImageFolder(data_dir, transform=transforms.Compose(
    [
        transforms.ToTensor(), # no need for the centercrop as it is at the cor
        transforms.CenterCrop(230),
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

evaluate_inverse_threshold = False
if args.exp_map_func == 'hard_inverse_threshold_explanation_map':
    evaluate_inverse_threshold = True
args.exp_map_func = eval(args.exp_map_func)

if args.ensemble_N >= 1 and args.eval_model_uncertainty:
    iou_loggers = [IOU_logger(0) for _ in range(args.ensemble_N)]
else:
    ad_logger = Average_Drop_logger(np.zeros((1,1)))
    ic_logger = Increase_Confidence_logger(np.zeros((1,1)))
    iou_logger = IOU_logger(0)
    ac_logger = Average_confidence_logger()

def evaluate_model_uncertainty(x, cam, args, annotations):
    centerCrop = transforms.CenterCrop(230)

    batch_cam_masks, per_member_logits = [], []
    for _ in range(args.ensemble_N):
        #batch-wise cam of the input size and scaled
        cam_targets = None
        cams, logit_scores = cam.forward(input_tensor=x, targets=cam_targets, retain_model_output=True)
        
        batch_cam_masks.append(threshold(cams))
        per_member_logits.append(logit_scores)
    
    # aggregate all annotation for each image into one single mask
    batch_aggregated_masks = []
    for per_img_annotation in annotations:
        loaded_npy_masks = [centerCrop(torch.tensor(np.load(a) // 255, device=Constants.DEVICE, dtype=torch.long)) 
                            for a in per_img_annotation] # convert to tensor and center crop
        aggregateds_mask = torch.sum(torch.stack(loaded_npy_masks, dim=0), dim=0)
        aggregateds_mask = aggregateds_mask.cpu().detach().numpy() if Constants.WORK_ENV == 'COLAB' else aggregateds_mask.detach().numpy()
        batch_aggregated_masks.append(aggregateds_mask)
    batch_aggregated_masks = np.array(np.stack(batch_aggregated_masks, axis=0), dtype=bool)

    # only take into account the correctly predicted images for each member of the ensemble
    batch_filtered_aggregated_masks = []
    for i, logit_scores in enumerate(per_member_logits):
        correct_predict_index = (torch.argmax(logit_scores, dim=1) == 1).cpu().detach().numpy() if Constants.WORK_ENV == 'COLAB' else (torch.argmax(logit_scores, dim=1) == 1).detach().numpy()
        batch_filtered_aggregated_masks.append(batch_aggregated_masks[correct_predict_index])
        batch_cam_masks[i] = batch_cam_masks[i][correct_predict_index] # replace the one that are only correctly classified

    # find the std and average iou
    for i, (batch_cam_mask, batch_aggregated_masks, iou_logger) in enumerate(zip(batch_cam_masks, batch_filtered_aggregated_masks, iou_loggers)):
        # Intersection over Union
        overlap = batch_cam_mask * batch_aggregated_masks
        union = batch_cam_mask + batch_aggregated_masks
        iou_logger.update(overlap.sum(), union.sum())
        print('member {}/{}; current overlap: {}; current union: {}; current IOU: {}'.format(i+1, args.ensemble_N, iou_logger.overlap, iou_logger.union, iou_logger.current_iou))

    return

def evaluate_segmentation_results(x, cams, args, annotations):
    """_summary_

    Args:
        model_wrapper (class): parent class of all "traditional" models
        cams (cams from pytroch-cam pacakage): 
        annotations(list of list): sublist represent all the annotations for an image
    """
    centerCrop = transforms.CenterCrop(230) # for cropping the data from loaded npy
    logit_scores = model_wrapper.model(x)
    batch_cam_mask = threshold(cams)

    # aggregate all annotation for each image into one single mask
    batch_aggregated_masks = []
    for per_img_annotation in annotations:
        loaded_npy_masks = [centerCrop(torch.tensor(np.load(a) // 255, device=Constants.DEVICE, dtype=torch.long)) 
                            for a in per_img_annotation] # convert to tensor and center crop
        aggregateds_mask = torch.sum(torch.stack(loaded_npy_masks, dim=0), dim=0)
        aggregateds_mask = aggregateds_mask.cpu().detach().numpy() if Constants.WORK_ENV == 'COLAB' else aggregateds_mask.detach().numpy()
        batch_aggregated_masks.append(aggregateds_mask)
    batch_aggregated_masks = np.array(np.stack(batch_aggregated_masks, axis=0), dtype=bool)

    # only take into account the correctly predicted images
    correct_predict_index = (torch.argmax(logit_scores, dim=1) == 1).cpu().detach().numpy() if Constants.WORK_ENV == 'COLAB' \
                             else (torch.argmax(logit_scores, dim=1) == 1).detach().numpy()
    batch_aggregated_masks = batch_aggregated_masks[correct_predict_index]
    batch_cam_mask = batch_cam_mask[correct_predict_index]

    # Intersection over Union
    overlap = batch_cam_mask * batch_aggregated_masks
    union = batch_cam_mask + batch_aggregated_masks
    iou_logger.update(overlap.sum(), union.sum())
    print('current iou: {}'.format(iou_logger.current_iou))


def evaluate_model_metrics(model_wrapper, grayscale_cam, args):
    print('Forward Passing the original images')
    Yci = model_wrapper.model(x)
    Yci = softmax(Yci, dim=1)
    Yci = Yci[range(Yci.shape[0]), y].unsqueeze(1) # get the score respects to the corresponding label
    
    print('Forward Passing the explanation images')
    img = denorm(x).detach().numpy() if Constants.WORK_ENV == 'LOCAL' else denorm(x).cpu().detach().numpy()
    grayscale_cam = np.expand_dims(grayscale_cam, 1)
    explanation_map = get_explanation_map(args.exp_map_func, img, grayscale_cam).to(device=Constants.DEVICE, dtype=Constants.DTYPE)
    exp_scores = model_wrapper.model(explanation_map)
    exp_scores = softmax(exp_scores, dim=1)
    Oci = exp_scores[range(Yci.shape[0]), y].unsqueeze(1)

    # collect metrics data
    Yci = Yci.detach().numpy() if Constants.WORK_ENV == 'LOCAL' else Yci.cpu().detach().numpy()
    Oci = Oci.detach().numpy() if Constants.WORK_ENV == 'LOCAL' else Oci.cpu().detach().numpy()
    if not evaluate_inverse_threshold:
        ad_logger.compute_and_update(Yci, Oci)
        ic_logger.compute_and_update(Yci, Oci)
        print('Progress: A.D: {}, I.C: {}'.format(ad_logger.current_metrics, ic_logger.current_metrics))
    else:
        ac_logger.compute_and_update(Yci, Oci)
        print('Progress: A.C {}'.format(ac_logger.current_metrics))


def generate_cams(x, args, image_order_book):
    """_summary_

    Args:
        x (tensor): assume denormed
        args (json): script arg input
        image_order_book (list of tuples): identify the image name
    """
    # denormalize the image NOTE: must be placed after forward passing
    # x = denorm(x)
    
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
        img = x[i, :].cpu().detach().numpy() if Constants.WORK_ENV == 'COLAB' else x[i, :].detach().numpy() 
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


for i, (x, y) in enumerate(dataloader):
    x = x.to(device=Constants.DEVICE, dtype=Constants.DTYPE)
    
    # NOTE: make sure i able index to the correct index
    print('--------- Forward Passing {}'.format(args.cam))
    # generate batch-wise cam
    # cam_targets = None
    # grayscale_cam = generate_cam_overlay(x, args, cam, cam_targets) #(batch, width, height)
    
    batch_annotations = [] # list of list of annotation
    if args.eval_model_uncertainty or args.eval_segmentation:
        img_names = [image_order_book[img_index + k][0].split('/')[-1] for k in range(x.shape[0])]
        for name in img_names:
            per_img_annotations = list(filter(lambda path: name[:-4] in path, annotation_file_list))
            per_img_annotations = [os.path.join(Constants.ANNOTATION_PATH, a) for a in per_img_annotations]
            batch_annotations.append(per_img_annotations)
    
    if args.eval_model_uncertainty:
        # cam_targets = None
        cam = switch_cam(args.cam, model_wrapper.model, model_target_layer)
        cam.model.train() # NOTE: VERY IMPORTANT STEP THE DROPOUT
        evaluate_model_uncertainty(x, cam, args, batch_annotations)
        img_index += x.shape[0]
    elif args.eval_segmentation:
        cam_targets = None
        grayscale_cam = generate_cam_overlay(x, args, cam, cam_targets) #(batch, width, height)
        
        evaluate_segmentation_results(x, grayscale_cam, args, batch_annotations)
        img_index += x.shape[0]
    elif args.run_mode == 'metrics':
        print('Forward Passing the original images')
        Yci = model_wrapper.model(x)
        Yci = softmax(Yci, dim=1)
        Yci = Yci[range(Yci.shape[0]), y].unsqueeze(1) # get the score respects to the corresponding label
        
        print('Forward Passing the explanation images')
        img = denorm(x).detach().numpy() if Constants.WORK_ENV == 'LOCAL' else denorm(x).cpu().detach().numpy()
        grayscale_cam = np.expand_dims(grayscale_cam, 1)
        explanation_map = get_explanation_map(args.exp_map_func, img, grayscale_cam).to(device=Constants.DEVICE, dtype=Constants.DTYPE)
        exp_scores = model_wrapper.model(explanation_map)
        exp_scores = softmax(exp_scores, dim=1)
        Oci = exp_scores[range(Yci.shape[0]), y].unsqueeze(1)

        # collect metrics data
        Yci = Yci.detach().numpy() if Constants.WORK_ENV == 'LOCAL' else Yci.cpu().detach().numpy()
        Oci = Oci.detach().numpy() if Constants.WORK_ENV == 'LOCAL' else Oci.cpu().detach().numpy()
        ad_logger.compute_and_update(Yci, Oci)
        ic_logger.compute_and_update(Yci, Oci)
        print('Progress: A.D: {}, I.C: {}'.format(ad_logger.current_metrics, ic_logger.current_metrics))

        # evaluate_model_metrics(model_wrapper, evaluate_model_metrics, args)

    else:
        generate_cams(denorm(x), args, image_order_book)
        # # denormalize the image NOTE: must be placed after forward passing
        # x = denorm(x)
        # print('--------- Generating {} Heatmap'.format(args.cam))
        # # for each image in a batch
        # for i in range(x.shape[0]):
        #     sample_name = image_order_book[img_index][0].split('/')[-1] # get the image name from the dataset

        #     # each image is a directory that contains all the experiment results
        #     dest = os.path.join(Constants.STORAGE_PATH, 'heatmaps', model_dir_name, '0' if y[i].item() == 0 else '1', sample_name)

        #     # save the original image in parallel
        #     if not os.path.exists(dest):
        #         os.makedirs(dest)
        #         # save the original image
        #         torchvision.utils.save_image(x[i, :], os.path.join(dest, 'original.jpg'))

        #     # swap the axis so that the show_cam_on_image works
        #     img = x[i, :].cpu().detach().numpy() if Constants.WORK_ENV == 'COLAB' else x[i, :].detach().numpy() 
        #     img = np.transpose(img, (1,2,0))

        #     # save the overlayed-attention map with the cam name as a tag
        #     attention_map = show_cam_on_image(img, grayscale_cam[i, :], use_rgb=True)
        #     cam_name = '{}-layers{}'.format(args.cam, args.layers)
            
        #     plt.ioff()

        #     logger = logging.getLogger()
        #     old_level = logger.level
        #     logger.setLevel(100)

        #     segmentation = plt.imshow(grayscale_cam[i, :], cmap='seismic')
        #     overlayed_image = plt.imshow(img, alpha=.5)
        #     plt.axis('off')
        #     plt.savefig(os.path.join(dest, cam_name+'_seismic.png'))

        #     segmented_image = img*threshold(grayscale_cam[i, :])[...,np.newaxis]
        #     segmented_image = np.where(segmented_image == 0, 100, segmented_image)
        #     segmented_image = plt.imshow(segmented_image)
        #     plt.axis('off')
        #     plt.savefig(os.path.join(dest, cam_name+'_segments.png'))
        #     plt.close()
            
        #     logger.setLevel(old_level)

        #     masked_img = Image.fromarray(attention_map, 'RGB')
        #     masked_img.save(os.path.join(dest, cam_name+'_rgb.jpg'))

        #     # update the sequential index for next iterations
        #     img_index += 1

# if args.eval_segmentation:
#     print('{}, IoU: {}'.format(args.layers, iou_logger.get_avg()))
# elif evaluate_inverse_threshold:
#     print('{};  Average Confidence: {}'.format(args.layers, ac_logger.get_avg()))
# elif args.run_mode == 'metrics':
#     print('{};  Average Drop: {}; Average IC: {}'.format(args.layers, ad_logger.get_avg(), ic_logger.get_avg()))
# else:
#     print('Done generating saliency map')

if args.eval_model_uncertainty:
    avg_iou = np.array([iou_logger.get_avg() for iou_logger in iou_loggers])
    print('{}, Avg IoU: {}, Std: {}'.format(args.target_layer, np.average(avg_iou), np.std(avg_iou)))
