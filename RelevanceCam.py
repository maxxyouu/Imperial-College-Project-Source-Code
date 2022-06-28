"""
From the selectiveKernelImplementation
"""

from ast import Str
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
# from ResNetLocal import resnet50, resnet34
from skresnet import skresnext50_32x4d
from layers import *
from RelevanceCamUtils import *
import Constants
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torchvision
from torch.utils.data.sampler import SequentialSampler
from Helper import denorm
import logging
from PIL import Image


my_parser = argparse.ArgumentParser(description='')
my_parser.add_argument('--model',
                        type=str, default='skresnext50_32x4d',
                        help='model to be used for training / testing') 
my_parser.add_argument('--pickle_name',
                        type=str, default='skresnext50_32x4d_supCon_last_3Layers',
                        help='pickel name for weight loading') 
my_parser.add_argument('--state_dict_path',
                        type=str, default=os.path.join(Constants.SAVED_MODEL_PATH, 'skresnext50_32x4d_supCon_last_1Layer'),
                        help='iteration for smoothing') 
my_parser.add_argument('--target_layer',
                        type=str, default='layer3',
                        help='cam layer for explanation') 
my_parser.add_argument('--batchSize',
                        type=int, default=256,
                        help='batch size to be used for training / testing') 
my_parser.add_argument('--lrpMode',
                        type=str, default='CLRP',
                        help='LRP mode for backpropgation')  
my_parser.add_argument('--alpha',
                        type=int, default=2,
                        help='alpha in the propagation rule')  
my_parser.add_argument('--dest_dir_name',
                        type=str, default='skresnext50_32x4d_pretrain',
                        help='destination folder in google drive')                    
args = my_parser.parse_args()


# SCRIPT PARAMETERS
# pt_name = 'skresnext50_32x4d_supCon_last_1layer' # fine tune with only one layer of the projection head in the classification model
pt_name = args.pickle_name# 'skresnext50_32x4d_supCon_last_3Layers'
pickel_name = './trained_models/{}'.format(pt_name) # NOTE: USE THE PRETRAIN ONE!
mode = args.target_layer # 'layer3'
LRP_MODE = args.lrpMode

# NOTE: setting for chosing alpha refers to https://arxiv.org/abs/1611.08191
CHOSEN_ALPHA = args.alpha # 2 # BETA = 1 # visually it is the best setting
target_layer = mode
target_class = None
cam_name = LRP_MODE + '-' + args.target_layer + '-' + str(args.alpha)

# create target result directory if not exists
# RESULT_FOLDER_BASE_NAME = './results' + '_' + pt_name + '_alpha' + str(CHOSEN_ALPHA) #LRP_MODE
# if not os.path.isdir(RESULT_FOLDER_BASE_NAME):
#     os.makedirs(RESULT_FOLDER_BASE_NAME)

model = skresnext50_32x4d(pretrained=False).eval()
model.num_classes = 2 #NOTE required to do CLRP and SGLRP
if 'simclr' in pickel_name or 'supCon' in pickel_name:
    if 'simclr' in pickel_name:
        cam_name = cam_name + '-simclr'
    else:
        cam_name = cam_name + '-supCon'
    dim_in = model.fc.in_features
    # the shape that can be placed with trained weights
    if '3Layers' in pt_name: # classification model that contains 3 projection head layers
        model.fc = Sequential(
                    Linear(dim_in, dim_in // 2),
                    ReLU(inplace=True),
                    Linear(dim_in // 2, dim_in // 4),
                    ReLU(inplace=True),
                    Linear(dim_in // 4, model.num_classes)
        )
        cam_name +=  '-width3'
    else:
        model.fc = Linear(model.fc.in_features, 2, device=torch.device('cpu')) 
        cam_name += '-width1'  
else:
    model.fc = Linear(model.fc.in_features, 2, device=torch.device('cpu'))
    cam_name += '-vanillaPretrain-width1' 

# load the trained weights
model.load_state_dict(torch.load('./{}.pt'.format(pickel_name), map_location=torch.device('cpu')))

if '3Layers' in pt_name:
    assert(model.fc[-1].out_features == 2)
else:
    assert(model.fc.out_features == 2)

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


data_dir = os.path.join(Constants.STORAGE_PATH, 'mutual_corrects')
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
dataloader = DataLoader(data, batch_size=2, sampler=sequentialSampler) # TODO: check image 18
image_order_book, img_index = data.imgs, 0

for x, y in dataloader:
    forward_handler = target_layer.register_forward_hook(forward_hook)
    backward_handler = target_layer.register_backward_hook(backward_hook)

    # NOTE: make sure i able index to the correct index
    print('--------- Forward Passing ------------')
    x = x.to(device=Constants.DEVICE, dtype=Constants.DTYPE)
    internal_R_cams, output = model(x, mode, [target_class], lrp=LRP_MODE, internal=False, alpha=CHOSEN_ALPHA)
    r_cams = internal_R_cams[0] # for each image in a batch

    # denormalize the image NOTE: must be placed after forward passing
    x = denorm(x)
    
    print('--------- Generating {} Heatmap'.format(args.cam))
    # for each image in a batch
    for i in range(x.shape[0]):
        
        sample_name = image_order_book[img_index][0].split('/')[-1] # get the image name from the dataset
        dest = os.path.join(Constants.STORAGE_PATH, 'heatmaps', args.dest_dir_name, '0' if y[i].item() == 0 else '1', sample_name)

        img = x[i, :].cpu().detach().numpy()
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 0, 1) #TODO check if you need this
        # img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img_show = cv2.resize(img_show,(230,230))

        # save the original image in parallel
        if not os.path.exists(dest):
            os.makedirs(dest)
            # save the original image
            torchvision.utils.save_image(x[i, :], os.path.join(dest, 'original.jpg'))

        plt.ioff()

        logger = logging.getLogger()
        old_level = logger.level
        logger.setLevel(100)

        segmentation = plt.imshow(r_cams[i, :], cmap='seismic')
        overlayed_image = plt.imshow(img, alpha=.5)
        plt.axis('off')
        plt.savefig(os.path.join(dest, cam_name+'_seismic.png'))

        segmented_image = img*threshold(r_cams[i, :])[...,np.newaxis]
        segmented_image = np.where(segmented_image == 0, 100, segmented_image) # with white color
        segmented_image = plt.imshow(segmented_image)
        plt.axis('off')
        plt.savefig(os.path.join(dest, cam_name+'_segments.png'))
        plt.close()
        
        logger.setLevel(old_level)

        # masked_img = Image.fromarray(attention_map, 'RGB')
        # masked_img.save(os.path.join(dest, LRP_MODE+'_rgb.jpg'))

        # update the sequential index for next iterations
        forward_handler.remove()
        backward_handler.remove()
        img_index += 1



# path_s = os.listdir('./picture')
# path_s.pop(path_s.index('.DS_Store'))

# for path in path_s:

#     if path == '.DS_Stovcre': continue
#     img_path_long = './picture/{}'.format(path)

#     # img_path_long = path
#     img = cv2.imread(img_path_long,1)
#     img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img_show = cv2.resize(img_show,(230,230))
#     img = np.float32(cv2.resize(img, (230,230)))/255

#     in_tensor = preprocess_image(img)#.cuda()
#     # R_CAM, output = model(in_tensor, lrp=lrp_mode, mode=mode, target_class=target_class)
#     internal_R_cams, output = model(in_tensor, mode, [target_class], lrp=LRP_MODE, internal=False, alpha=CHOSEN_ALPHA)

#     for i, cam in enumerate(internal_R_cams):
#         fig = plt.figure(figsize=(10, 10))
#         plt.subplots_adjust(bottom=0.01)

#         plt.subplot(2, 5, 1)
#         plt.imshow(img_show)
#         plt.title('Original')
#         plt.axis('off')

#         plt.subplot(2, 5, 1 + 5)
#         plt.imshow(img_show)
#         plt.axis('off')

#         print('LRP1')
#         plt.subplot(2, 5, 5)
#         cam = cam.reshape(size,size).detach().numpy()
#         cam = cv2.resize(cam, (230, 230))
#         plt.imshow((cam),cmap='seismic')
#         plt.imshow(img_show, alpha=.5)
#         plt.title('Relevance_CAM', fontsize=15)
#         plt.axis('off')

#         print('LRP2')
#         plt.subplot(2, 5, 5 + 5)
#         plt.imshow(img_show*threshold(cam)[...,np.newaxis])
#         plt.title('Relevance_CAM', fontsize=15)
#         plt.axis('off')

#         plt.savefig('./{}/{}-{}{}-{}.jpg'.format(RESULT_FOLDER_BASE_NAME, path[:-4], mode, i, LRP_MODE))

#         # plt.clf()
#         plt.close()

#     forward_handler.remove()
#     backward_handler.remove()
# print('Done')   