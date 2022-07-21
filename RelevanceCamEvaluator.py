"""
Script to compute the necessary metrics of a model
A.D, A.I, Modified A.I, and segmentation
TODO: Implement AD, AI,
Target layer to be used is accor
"""

from copy import deepcopy
import torch
from torch.nn.functional import softmax
import os
from ResNetLocal import resnet50
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
from PIL import Image
from resnet import resnet50 as lrp_resnet50
from vgg import vgg11_bn as lrp_vgg11_bn

default_model_name = 'skresnext50_32x4d'
my_parser = argparse.ArgumentParser(description='')
my_parser.add_argument('--model_name',
                        type=str, default=default_model_name,
                        help='model name to be used for model retrival and weight replacement') 
my_parser.add_argument('--model_weights',
                        type=str, default=os.path.join(Constants.SAVED_MODEL_PATH, default_model_name+'_pretrain.pt'),
                        help='Destination for the model weights') 
my_parser.add_argument('--target_layer',
                        type=str, default='layer3,layer4',
                        help='sample: 3,4') 
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
my_parser.add_argument('--data_location',
                        type=str, default=Constants.ANNOTATED_IMG_PATH, #os.path.join(Constants.STORAGE_PATH, 'mutual_corrects'), # for segmentation: Constants.ANNOTATED_IMG_PATH
                        help='data directory')   
my_parser.add_argument('--alpha',
                        type=float, default=2, # example: ckpt_epoch_500
                        help='alpha for relevance cam')          
my_parser.add_argument('--eval_segmentation',
                        type=bool, action=argparse.BooleanOptionalAction,
                        help='true indicate evaluate the segmentation performance of the cam method')
my_parser.add_argument('--annotation_path',
                        type=str, default=Constants.ANNOTATION_PATH,
                        help='path for the imge annotation')    
my_parser.add_argument('--headWidth',
                        type=int, default=1,
                        help='true indicate evaluate the segmentation performance of the cam method')
my_parser.add_argument('--plusplusMode',
                        type=bool, action=argparse.BooleanOptionalAction,
                        help='using gradcam plusplus mode')              
args = my_parser.parse_args()

# Sanity checks for the script arguments
print('Model Name: {}'.format(args.model_name))
print('Model Weight Destination: {}'.format(args.model_weights))
print('Target Layer: {}'.format(args.target_layer))
print('Batch Size: {}'.format(args.batch_size))
if args.evaluate_all_layers is None:
    args.evaluate_all_layers = False
print('Unbias Layer Selection: {}'.format(args.evaluate_all_layers))
print('Explanation map style: {}'.format(args.exp_map_func))
print('CAM: {}'.format(args.cam))
print('Alpha: {}'.format(args.alpha))
print('Data Location {}'.format(args.data_location))
if args.plusplusMode is None:
    args.plusplusMode = False # for local debug only
print('Plus Plus Mode: {}'.format(args.plusplusMode))
print('Head Width: {}'.format(args.headWidth))


if Constants.WORK_ENV == 'LOCAL': # NOTE: FOR DEBUG PURPOSE
    args.eval_segmentation = True 
if args.eval_segmentation is None or args.eval_segmentation == False:
    args.eval_segmentation = False
else:
    #make sure in the correct data source location
    assert(args.data_location == Constants.ANNOTATED_IMG_PATH)
    assert(args.annotation_path == Constants.ANNOTATION_PATH)
    annotation_file_list = os.listdir(args.annotation_path)
print('Evaluate Segmentation {}'.format(args.eval_segmentation))
data_dir = args.data_location

if args.model_name == 'resnet50':
    model = lrp_resnet50(pretrained=False)
elif args.model_name == 'vgg11_bn':
    model = lrp_vgg11_bn(pretrained=False)
else:
    model = skresnext50_32x4d(pretrained=False)
model.num_classes = 2 #NOTE required to do CLRP and SGLRP

# handle the projection head
headWidth = args.headWidth
assert(headWidth > 0 and headWidth <= 3)

if 'vgg' in args.model_name:
    if headWidth == 1:
        model.classifier = Linear(model.classifier[0].in_features, model.num_classes, device=Constants.DEVICE, dtype=Constants.DTYPE)
    elif headWidth == 2:
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[0].in_features, model.classifier[0].out_features, device=Constants.DEVICE, dtype=Constants.DTYPE),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(model.classifier[0].out_features, model.num_classes, device=Constants.DEVICE, dtype=Constants.DTYPE),  
        )
    elif headWidth == 3:
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[0].in_features, model.classifier[0].out_features, device=Constants.DEVICE, dtype=Constants.DTYPE),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(model.classifier[3].in_features, model.classifier[3].out_features, device=Constants.DEVICE, dtype=Constants.DTYPE),  
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(model.classifier[3].out_features, model.num_classes, device=Constants.DEVICE, dtype=Constants.DTYPE), 
        )    
else: # other models
    if headWidth == 1:
        model.fc = Linear(model.fc.in_features, model.num_classes, device=Constants.DEVICE, dtype=Constants.DTYPE)
    elif headWidth == 2:
        model.fc = Sequential(*[
            Linear(model.fc.in_features, model.fc.in_features // 2, device=Constants.DEVICE, dtype=Constants.DTYPE),
            ReLU(),
            Dropout(),
            Linear(model.fc.in_features // 2, model.num_classes, device=Constants.DEVICE, dtype=Constants.DTYPE)
        ])
    elif headWidth == 3:
        model.fc = Sequential(*[
            Linear(model.fc.in_features, model.fc.in_features // 2, device=Constants.DEVICE, dtype=Constants.DTYPE),
            ReLU(),
            Dropout(),
            Linear(model.fc.in_features // 2, model.fc.in_features // 4, device=Constants.DEVICE, dtype=Constants.DTYPE),
            ReLU(),
            Dropout(),
            Linear(model.fc.in_features // 4, model.num_classes, device=Constants.DEVICE, dtype=Constants.DTYPE)
        ])
# load the trained weights
model.load_state_dict(torch.load(args.model_weights, map_location=Constants.DEVICE))
model.to(Constants.DEVICE)
model.eval() # after loading the model, put the model into evaluation mode
print('Model successfully loaded')

aggregation = False
target_layer = args.target_layer
if target_layer == 'layer2':
    target_layer = model.layer2
elif target_layer == 'layer3':
    target_layer = model.layer3
elif target_layer == 'layer4':
    target_layer = model.layer4
elif target_layer == 'layer1':
    target_layer = model.layer1
else: # layer aggregation case
    # mode all and use the appropriate one
    aggregation = True

value = dict()
def forward_hook(module, input, output):
    value['activations'] = output
def backward_hook(module, input, output):
    value['gradients'] = output[0]

data_transformers = transforms.Compose(
    [
        transforms.ToTensor(), # no need for the centercrop as it is at the cor
        transforms.CenterCrop(230),
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


if not aggregation:
    forward_handler = target_layer.register_forward_hook(forward_hook)
    backward_handler = target_layer.register_full_backward_hook(backward_hook)
else:
    layers = args.target_layer.split(',')
    fhs, bhs = [], []
    for layer in layers:
        fh = getattr(model, layer).register_forward_hook(forward_hook)
        bh = getattr(model, layer).register_full_backward_hook(backward_hook)
        fhs.append(fh)
        bhs.append(bh)

print('Registered Hooks')

evaluate_inverse_threshold = False
if args.exp_map_func == 'hard_inverse_threshold_explanation_map':
    evaluate_inverse_threshold = True
args.exp_map_func = eval(args.exp_map_func)


if args.evaluate_all_layers:
    ad_logger = Average_Drop_logger(np.zeros((1, 4)))
    ic_logger = Increase_Confidence_logger(np.zeros((1, 4)))
else:
    ad_logger = Average_Drop_logger(np.zeros((1,1)))
    ic_logger = Increase_Confidence_logger(np.zeros((1,1)))
    iou_logger = IOU_logger(0)
    ac_logger = Average_confidence_logger()


def evaluate_model_metrics(x, args):
    Yci = None
    if args.evaluate_all_layers:
        layer_explanations = [] # each index location store a batch-size of cam explanation map
        for layer in layers:
            cams, Yci = model(x, mode=layer,  target_class=[None], plusplusMode=args.plusplusMode,  alpha=args.alpha)
            Yci = softmax(Yci, dim=1)
            layer_explanations.append(resize_cam(cams[0]))
            Yci = Yci[range(Yci.shape[0]), y].unsqueeze(1) # only care about the score for the true label
        cam = layer_explanations[layer_idx_mapper[args.target_layer]] # retrieve the target layer according to the argument provided for the following code
    else:
        cams, Yci = model(x, mode=args.target_layer, target_class=[None], plusplusMode=args.plusplusMode,  alpha=args.alpha)
        Yci = softmax(Yci, dim=1)
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
            explanation_map = get_explanation_map(args.exp_map_func, img, cam).to(device=Constants.DEVICE, dtype=Constants.DTYPE)

            ## NOTE: FOR DEBUG
            _, exp_scores = model(explanation_map, mode='output', target_class=[None], plusplusMode=args.plusplusMode,  alpha=args.alpha)
            exp_scores = softmax(exp_scores, dim=1)
            layer_explanation_scores.append(exp_scores[range(Yci.shape[0]), y]) # the corresponding label score (the anchor)

        Oci = torch.stack(layer_explanation_scores, dim=1)

    else:
        explanation_map = get_explanation_map(args.exp_map_func, img, cam).to(device=Constants.DEVICE, dtype=Constants.DTYPE)
        _, exp_scores = model(explanation_map, mode=args.target_layer, target_class=[None], plusplusMode=args.plusplusMode,  alpha=args.alpha)
        exp_scores = softmax(exp_scores, dim=1)
        Oci = exp_scores[range(Yci.shape[0]), y].unsqueeze(1)
        # compare the explanation score with the original score

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


def evaluate_segmentation_metrics(x, annotations, args):
    """using intersectin over union as a metric to evaluate the segmentation performance

    Args:
        x (tensor): tensor in a device
        img_name (_type_): the image with suffix
        annotations (list of list): list of annotation paths (can be more than 1 annoataion for the same imge)
        args (dicts): user input for the script
    """
    centerCrop = transforms.CenterCrop(230)
    r_cam, logit_scores = model(x, mode=args.target_layer, target_class=[None], plusplusMode=args.plusplusMode,  alpha=args.alpha)
    if aggregation:
        cams = [resize_cam(map) for map in r_cam]
        # aggregate across the cam axis by performing elemenwise max ops and return a single cam object
        cam = np.amax(np.stack(cams, axis=0), axis=0)
    else: 
        cam = resize_cam(r_cam[0]) # [batch_size, width, heigh]
    batch_cam_mask = threshold(cam).squeeze(1)
    # plt.imshow(cam[0,:].squeeze(0), cmap='seismic')
    # plt.imshow(np.transpose(denorm(x[0,:]), (1,2,0)), alpha=.5)
    #NOTE: we might want to igore the one that is wrongly classified.
    batch_aggregated_masks = []
    for per_img_annotation in annotations:
        loaded_npy_masks = [centerCrop(torch.tensor(np.load(a) // 255, device=Constants.DEVICE, dtype=torch.long)) for a in per_img_annotation] # convert to tensor and center crop
        aggregateds_mask = torch.sum(torch.stack(loaded_npy_masks, dim=0), dim=0)
        aggregateds_mask = aggregateds_mask.cpu().detach().numpy() if Constants.WORK_ENV == 'COLAB' else aggregateds_mask.detach().numpy()
        # plt.imshow(aggregateds_mask)
        # plt.imshow(batch_cam_mask[0,:])
        batch_aggregated_masks.append(aggregateds_mask)

    batch_aggregated_masks = np.array(np.stack(batch_aggregated_masks, axis=0), dtype=bool)
    
    # only take into account the correctly predicted images
    correct_predict_index = (torch.argmax(logit_scores, dim=1) == 1).cpu().detach().numpy() if Constants.WORK_ENV == 'COLAB' else (torch.argmax(logit_scores, dim=1) == 1).detach().numpy()
    batch_aggregated_masks = batch_aggregated_masks[correct_predict_index]
    batch_cam_mask = batch_cam_mask[correct_predict_index]
    
    # Intersection over Union
    overlap = batch_cam_mask * batch_aggregated_masks
    union = batch_cam_mask + batch_aggregated_masks
    iou_logger.update(overlap.sum(), union.sum())
    print('current iou: {}'.format(iou_logger.current_iou))

for i, (x, y) in enumerate(dataloader):
    # NOTE: make sure i able index to the correct index
    print('--------- Forward Passing the Original Data ------------')
    x = x.to(device=Constants.DEVICE, dtype=Constants.DTYPE)
    if args.eval_segmentation:
        # get the segmentation annotations using the img names in a batch

        img_names = [image_order_book[img_index + k][0].split('/')[-1] for k in range(x.shape[0])]
        batch_annotations = [] # list of list of annotation
        for name in img_names:
            per_img_annotations = list(filter(lambda path: name[:-4] in path, annotation_file_list))
            per_img_annotations = [os.path.join(Constants.ANNOTATION_PATH, a) for a in per_img_annotations]
            batch_annotations.append(per_img_annotations)
        evaluate_segmentation_metrics(x, batch_annotations, args)
    else:
        evaluate_model_metrics(x, args)
    
    img_index += x.shape[0]

if not aggregation:
    forward_handler.remove()
    backward_handler.remove()
else:
    for fh, bh in zip(fhs, bhs):
        fh.remove()
        bh.remove()

# print the metrics results
if not args.eval_segmentation and not evaluate_inverse_threshold:
    print('{};  Average Drop: {}; Average IC: {}'.format(args.target_layer, ad_logger.get_avg(), ic_logger.get_avg()))
elif evaluate_inverse_threshold:
    print('{};  Average Confidence: {}'.format(args.target_layer, ac_logger.get_avg()))
else:
    print('{}, IoU: {}'.format(args.target_layer, iou_logger.get_avg()))

# for j in range(cam.shape[0]):
#     plt.imshow(cam[j,:].squeeze(0), cmap='seismic')
#     plt.imshow(np.transpose(img[j,:], (1,2,0)), alpha=.5)
#     plt.axis('off')
