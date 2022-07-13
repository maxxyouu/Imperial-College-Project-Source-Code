from abc import abstractmethod
from typing import Callable
import Constants
import numpy as np
import cv2
from Helper import max_min_lrp_normalize
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
    cam = max_min_lrp_normalize(cam) # normalize the saliency results
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

def hard_inverse_threshold_explanation_map(img, cam):
    """
    used for select which layer of the relevance cam to be used
    """
    explanation_map = img*threshold(cam, inverse=True)
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
    """
    Args:
        exp_map (Callable): either hard_threshold_explanation_map or soft_explanation_map
        img (tensor): _description_
        cam (tensor): _description_
    """
    # explanation_map = img*threshold(cam)
    explanation_map = exp_map(img, cam)
    explanation_map = torch.from_numpy(explanation_map)
    for i in range(explanation_map.shape[0]):
        inplace_normalize(explanation_map[i, :])
    return explanation_map.requires_grad_(True)


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

class IOU_logger:
    """
    The higher the better
    """
    def __init__(self, metrics_initial) -> None:
        self.overlap = 0
        self.union = 0
        self.current_iou = 0
    def get_avg(self):
        return self.overlap / self.union

    def update(self, overlap, union):
        self.current_iou = overlap / union
        self.overlap += overlap
        self.union += union

class Average_Drop_logger(metrics_logger):
    """
    The lower the better
    """
    def __init__(self, metrics_initial) -> None:
        super().__init__(metrics_initial)

    def get_avg(self):
        return self.metrics * 100 / self.N # if using softmax score, x100


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
    """The higher the better"""
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
        return self.metrics * 100 / self.N # if using softmax score, x100


class Average_confidence_logger():
    """ 
    Refer to the layerCAM paper table VI
    must be companied by hard inverse threshold
    """
    def __init__(self) -> None:
        self.true_score_avg = 0
        self.occluded_score_avg = 0
        self.N = 0
    
    def compute_and_update(self, Yci, Oci):
        self.true_score_avg += (Yci * 100)
        self.occluded_score_avg += (Oci * 100)
        assert(Yci.shape == Oci.shape)
        self.N += Yci.shape[0]
        

    def get_avg(self):
        return (self.true_score_avg / self.N, self.occluded_score_avg / self.N) # if using softmax score, x100


