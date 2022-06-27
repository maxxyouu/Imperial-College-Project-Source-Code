"""
From the selectiveKernelImplementation repo
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
# from ResNetLocal import resnet50, resnet34
from skresnet import skresnext50_32x4d
from layers import *
from RelevanceCamUtils import *


# SCRIPT PARAMETERS
# pt_name = 'skresnext50_32x4d_supCon_last_1layer' # fine tune with only one layer of the projection head in the classification model
pt_name = 'skresnext50_32x4d_supCon_last_3Layers'
pickel_name = './trained_models/{}'.format(pt_name) # NOTE: USE THE PRETRAIN ONE!
mode = 'layer3'
LRP_MODE ='CLRP'

# NOTE: setting for chosing alpha refers to https://arxiv.org/abs/1611.08191
CHOSEN_ALPHA = 2 # BETA = 1 # visually it is the best setting
target_layer = mode
target_class = None


# create target result directory if not exists
RESULT_FOLDER_BASE_NAME = './results' + '_' + pt_name + '_alpha' + str(CHOSEN_ALPHA) #LRP_MODE
if not os.path.isdir(RESULT_FOLDER_BASE_NAME):
    os.makedirs(RESULT_FOLDER_BASE_NAME)

model = skresnext50_32x4d(pretrained=False).eval()
model.num_classes = 2 #NOTE required to do CLRP and SGLRP
if 'simclr' in pickel_name or 'supCon' in pickel_name:
    dim_in = model.fc.in_features
    # the shape that can be placed with trained weights
    if '3layers' in pt_name: # classification model that contains 3 projection head layers
        model.fc = Sequential(
                    Linear(dim_in, dim_in // 2),
                    ReLU(inplace=True),
                    Linear(dim_in // 2, dim_in // 4),
                    ReLU(inplace=True),
                    Linear(dim_in // 4, model.num_classes)
        )
    else:
        model.fc = Linear(model.fc.in_features, 2, device=torch.device('cpu'))   
else:
    model.fc = Linear(model.fc.in_features, 2, device=torch.device('cpu'))
# load the trained weights
model.load_state_dict(torch.load('./{}.pt'.format(pickel_name), map_location=torch.device('cpu')))

if '3layers' in pt_name:
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


path_s = os.listdir('./picture')
path_s.pop(path_s.index('.DS_Store'))

for path in path_s:

    if path == '.DS_Stovcre': continue
    img_path_long = './picture/{}'.format(path)
    forward_handler = target_layer.register_forward_hook(forward_hook)
    backward_handler = target_layer.register_backward_hook(backward_hook)
    # img_path_long = path
    img = cv2.imread(img_path_long,1)
    img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_show = cv2.resize(img_show,(230,230))
    img = np.float32(cv2.resize(img, (230,230)))/255

    in_tensor = preprocess_image(img)#.cuda()
    # R_CAM, output = model(in_tensor, lrp=lrp_mode, mode=mode, target_class=target_class)
    internal_R_cams, output = model(in_tensor, mode, [target_class], lrp=LRP_MODE, internal=False, alpha=CHOSEN_ALPHA)

    for i, cam in enumerate(internal_R_cams):
        fig = plt.figure(figsize=(10, 10))
        plt.subplots_adjust(bottom=0.01)

        plt.subplot(2, 5, 1)
        plt.imshow(img_show)
        plt.title('Original')
        plt.axis('off')

        plt.subplot(2, 5, 1 + 5)
        plt.imshow(img_show)
        plt.axis('off')

        print('LRP1')
        plt.subplot(2, 5, 5)
        cam = cam.reshape(size,size).detach().numpy()
        cam = cv2.resize(cam, (230, 230))
        plt.imshow((cam),cmap='seismic')
        plt.imshow(img_show, alpha=.5)
        plt.title('Relevance_CAM', fontsize=15)
        plt.axis('off')

        print('LRP2')
        plt.subplot(2, 5, 5 + 5)
        plt.imshow(img_show*threshold(cam)[...,np.newaxis])
        plt.title('Relevance_CAM', fontsize=15)
        plt.axis('off')

        plt.savefig('./{}/{}-{}{}-{}.jpg'.format(RESULT_FOLDER_BASE_NAME, path[:-4], mode, i, LRP_MODE))

        # plt.clf()
        plt.close()

    forward_handler.remove()
    backward_handler.remove()
print('Done')   