from torchvision import transforms, datasets
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad

import numpy as np
import os
import torchvision
from PIL import Image

from Helper import denorm, switch_cam

# local imports
from BaselineModel import Pytorch_default_resNet, Pytorch_default_resnext, Pytorch_default_skres, Pytorch_default_skresnext, Pytorch_default_vgg
import Constants

class CAM_Generator:
    def __init__(self, model_name, model_wrapper, target_layers, data, cams) -> None:
        self.cams = cams
        self.model_name = model_name
        self.model_wrapper = model_wrapper
        self.model_wrapper.load_learned_weights('./trained_models/{}.pt'.format(model_name))
        self.target_layers = target_layers
        # make sure the self.x contains al lthe data within target_layers
        dataloader = DataLoader(data, batch_size=len(data))
        self.x, _ = next(iter(dataloader))

    def create_heatmap_dest(self, i, feature):
        dest = os.path.join(Constants.STORAGE_PATH, 'heatmaps', self.model_name, 'image-{}'.format(i))
        if not os.path.exists(dest):
            os.makedirs(dest)
            # save the original image
            torchvision.utils.save_image(feature, os.path.join(dest, 'original.jpg')) 

    def create_heat_mask_and_save(self, dest, img, feature_mask, cam_name):
        attention_map = show_cam_on_image(img, feature_mask, use_rgb=True)
        masked_img = Image.fromarray(attention_map, 'RGB')
        masked_img.save(os.path.join(dest, '{}.jpg'.format(cam_name)))

    def generate_cam(self):
        # for each image, it has a folder that store all the cam heatmaps
        for cam_name in self.cams:
            # make sure the cam is freed after used
            # NOTE: otherwise, odd results will be formed
            with switch_cam(cam_name, resnet18.model, [resnet18_target_layer]) as cam: 
                print('--------- Forward Passing {}'.format(cam_name))
                grayscale_cam = cam(input_tensor=x, targets=None)
                
                # denormalize the image NOTE: must be placed after forward passing
                # x.mul_(Constants.DATA_STD).add_(Constants.DATA_MEAN)
                x = denorm(x)
                
                print('--------- Generating CAM')
                # for each image in a batch
                for i in range(x.shape[0]):
                    # create directory this image-i if needed
                    self.create_heatmap_dest(i, x[i, :])

                    # swap the axis so that the show_cam_on_image works NOTE: otherwise wont work
                    img = x[i, :].cpu().detach().numpy()
                    img = np.swapaxes(img, 0, 2)
                    img = np.swapaxes(img, 0, 1)

                    # save the overlayed-attention map with the cam name as a tag
                    self.create_heat_mask_and_save(dest, img, grayscale_cam[i, :], cam_name)
                    # attention_map = show_cam_on_image(img, grayscale_cam[i, :], use_rgb=True)
                    # masked_img = Image.fromarray(attention_map, 'RGB')
                    # masked_img.save(os.path.join(dest, '{}.jpg'.format(cam_name)))
        

if __name__ == '__main__':

    smoothing = True
    
    # model_name = 'resnet18'
    # model_wrapper = Pytorch_default_resNet(model_name=model_name)
    # model.load_learned_weights('./trained_models/{}.pt'.format(model_name))
    # model_target_layer = [resnet18.model.layer4[-1]]

    # model_name = 'resnet18_pretrain'
    # model_wrapper = Pytorch_default_resNet(model_name='resnet18')
    # model_wrapper.load_learned_weights('./trained_models/{}.pt'.format(model_name))
    # model_target_layer = [model_wrapper.model.layer4[-1]]

    # model_name = 'skresnet18'
    # model_wrapper = Pytorch_default_skres(model_name=model_name)
    # model_wrapper.load_learned_weights('./trained_models/{}.pt'.format(model_name))
    # model_target_layer = [ model_wrapper.model.layer4[-1]]

    # model_name = 'resnext50_32x4d_pretrain'
    # model_wrapper = Pytorch_default_resnext(model_name='resnext50_32x4d')
    # model_wrapper.load_learned_weights('./trained_models/{}.pt'.format(model_name))
    # model_target_layer = [model_wrapper.model.layer4[-1]]

    model_name = 'skresnext50_32x4d_pretrain'
    model_wrapper = Pytorch_default_skresnext(model_name='skresnext50_32x4d')
    model_wrapper.load_learned_weights('./trained_models/{}.pt'.format(model_name))
    model_target_layer = [model_wrapper.model.layer4[-1]]

    # model_name = 'skresnext50_32x4d'
    # model_wrapper = Pytorch_default_skresnext(model_name='skresnext50_32x4d')
    # model_wrapper.load_learned_weights('./trained_models/{}.pt'.format(model_name))
    # model_target_layer = [model_wrapper.model.layer4[-1]]

    # NOTE: to load the pretrain model, the base model must come from the the pytorch NOT timm
    # model_name = 'vgg11_bn_pretrain'
    # model_wrapper = Pytorch_default_vgg(model_name='vgg11_bn')
    # model_wrapper.load_learned_weights('./trained_models/{}.pt'.format(model_name))
    # model_target_layer = [model_wrapper.model.features[-1]]

    model_dir_name = model_name + '_noSmooth' if not smoothing  else model_name
    data = datasets.ImageFolder('./correct_preds', transform=transforms.Compose(
        [
            transforms.ToTensor(), # no need for the centercrop as it is at the cor
            transforms.Normalize(
                [Constants.DATA_MEAN, Constants.DATA_MEAN, Constants.DATA_MEAN], 
                [Constants.DATA_STD,Constants.DATA_STD, Constants.DATA_STD])
        ]
    ))

    # for each image, it has a folder that store all the cam heatmaps
    dataloader = DataLoader(data, batch_size=48)
    x, _ = next(iter(dataloader))

    cams = ['gradcam++'] # 'scorecam', 'ablationcam', 'xgradcam', 'eigencam',
    for cam_name in cams:
        # make sure the cam is freed after used
        # NOTE: otherwise, odd results will be formed
        with switch_cam(cam_name, model_wrapper.model, model_target_layer) as cam: 
            print('--------- Forward Passing {}'.format(cam_name))
            grayscale_cam = cam(input_tensor=x, targets=None, aug_smooth=smoothing)
            
            # denormalize the image NOTE: must be placed after forward passing
            x = denorm(x)
            
            print('--------- Generating CAM')
            # for each image in a batch
            for i in range(x.shape[0]):
                # create directory this image-i if needed
                dest = os.path.join(Constants.STORAGE_PATH, 'heatmaps', model_dir_name, 'image-{}'.format(i))
                if not os.path.exists(dest):
                    os.makedirs(dest)
                    # save the original image
                    torchvision.utils.save_image(x[i, :], os.path.join(dest, 'original.jpg'))

                # swap the axis so that the show_cam_on_image works
                img = x[i, :].cpu().detach().numpy()
                img = np.swapaxes(img, 0, 2)
                img = np.swapaxes(img, 0, 1)

                # save the overlayed-attention map with the cam name as a tag
                attention_map = show_cam_on_image(img, grayscale_cam[i, :], use_rgb=True)
                masked_img = Image.fromarray(attention_map, 'RGB')
                masked_img.save(os.path.join(dest, '{}.jpg'.format(cam_name)))