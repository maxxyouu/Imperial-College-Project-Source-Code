from typing import Callable
import Constants
import numpy as np
import cv2
from RelevanceCamUtils import *
from torchvision import transforms


def resize_cam(cam):
    if Constants.WORK_ENV == 'COLAB':
        cam = cam.cpu().detach().numpy()
    else:
        cam = cam.detach().numpy()

    # resize each image and stack them back
    cam = np.stack(
        [cv2.resize(cam[i, :].squeeze(0), (230, 230)) for i in range(cam.shape[0])],
        axis=0)
    cam = np.expand_dims(cam, axis=1)
    return cam

def resize_img(img):
    """assume img is denormed and a copy"""
    # x = denorm(x)
    # img = deepcopy(x)
    if Constants.WORK_ENV == 'COLAB':
        img = img.cpu().detach().numpy()
    else:
        img = img.detach().numpy()
    
    # transform back to (2, 0, 1) so that the normalization works
    img = np.stack(
        [np.transpose(cv2.cvtColor(np.transpose(img[i, :], (1, 2, 0)), cv2.COLOR_BGR2RGB), (2, 0, 1)) for i in range(img.shape[0])],
        axis=0)

    return img

def hard_threshold_explanation_map(img, cam):
    """
    used for select which layer of the relevance cam to be used
    """
    explanation_map = img*threshold(cam)
    return explanation_map

def soft_explanation_map(img, cam):
    """in the grad cam paper
    used for examine the metrics AD, AI
    """
    return img * np.maximum(cam, 0)

    
inplace_normalize = transforms.Normalize(
            [Constants.DATA_MEAN, Constants.DATA_MEAN, Constants.DATA_MEAN], 
            [Constants.DATA_STD,Constants.DATA_STD, Constants.DATA_STD], inplace=True)

def get_explanation_map(exp_map: Callable, img, cam):
    # explanation_map = img*threshold(cam)
    explanation_map = exp_map(img, cam)
    explanation_map = torch.from_numpy(explanation_map)
    for i in range(explanation_map.shape[0]):
        inplace_normalize(explanation_map[i, :])
    return explanation_map

def A_D():
    pass

def A_I():
    pass

def m_A_I():
    pass

