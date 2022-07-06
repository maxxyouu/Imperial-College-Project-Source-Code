from abc import abstractmethod
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
    return explanation_map.requires_grad_(True)

def A_D():
    pass

def A_I():
    pass

def m_A_I():
    pass


class metrics_logger:
    def __init__(self, metrics_initial) -> None:
        self.metrics = metrics_initial
        self.N = 0
        self.current_metrics = metrics_initial

    def update(self, current, n):
        self.N += n
        self.metrics += current

    @abstractmethod
    def get_avg(self):
        return NotImplementedError('metrics specific function, to be implemented')

    @abstractmethod
    def compute_and_update(self, Yci, Oci):
        """Specific to a metrics"""
        return NotImplementedError('metrics specific function, to be implemented')

class Average_Drop_logger(metrics_logger):
    def __init__(self, metrics_initial) -> None:
        super().__init__(metrics_initial)

    def get_avg(self):
        return self.metrics

    def compute_and_update(self, Yci, Oci):
        """metrics specific

        Args:
            Yci (numpy array): score for the original image(s)
            Oci (numpy array): score for the explanation map(s)
            assume Yci and Oci are of the same shape
        """

        # batch-wise percentage drop
        percentage_drop = (Yci - Oci) / Yci
        percentage_drop = np.maximum(percentage_drop, 0)
        
        # aggregate the batch statistics
        batch_size = percentage_drop.shape[0]
        batch_pd = np.sum(percentage_drop, axis=0)
        self.current_metrics = batch_pd
        super().update(batch_pd, batch_size)

class Increase_Confidence_logger(metrics_logger):
    def __init__(self, metrics_initial) -> None:
        super().__init__(metrics_initial)

    def compute_and_update(self, Yci, Oci):
        """metrics specific

        Args:
            Yci (numpy array): score for the original image(s)
            Oci (numpy array): score for the explanation map(s)
            assume Yci and Oci are of the same shape
        """
        indicator = Yci < Oci
        batch_size = indicator.shape[0]
        # aggregate the batch statistics    
        increase_in_confidence = np.sum(indicator, axis=0)
        self.current_metrics = increase_in_confidence
        super().update(increase_in_confidence, batch_size)

    def get_avg(self):
        return self.metrics / self.N

