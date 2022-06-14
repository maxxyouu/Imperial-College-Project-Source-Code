from torchvision import transforms, datasets
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.utils.data import DataLoader
# from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np
import os
import torchvision
import torch
from PIL import Image

from Helper import denorm, switch_cam, extract_attention_cam_args, get_trained_model

# local imports
from BaselineModel import Pytorch_default_resNet, Pytorch_default_resnext, Pytorch_default_skres, Pytorch_default_skresnext, Pytorch_default_vgg
import Constants
        
def add_noise(x, noise_level):
    # noise = np.random.normal(0.0, scale=(noise_level / torch.max(x_reshaped, 1)[0] - torch.min(x_reshaped, 1)[0]))
    # x_reshaped = torch.reshape(x, (x.shape[0], -1))
    noise = np.random.normal(0.0, scale=noise_level)
    noise = torch.tensor(noise, device=Constants.DEVICE, dtype=Constants.DTYPE)
    return x + noise

def define_model_dir_path(args):
    model_dir_name = args.model + '_noiseSmooth' if args.noiseSmooth else args.model
    if args.noiseSmooth:
        model_dir_name += '_noise{}_iters{}'.format(args.std, args.iterations)
    return model_dir_name

if __name__ == '__main__':
    args = extract_attention_cam_args()

    model_wrapper = get_trained_model(args.model)
    model_target_layer = [model_wrapper.model.layer3[-1], model_wrapper.model.layer4[-1]]

    model_dir_name = define_model_dir_path(args)

    data_dir = os.path.join(Constants.STORAGE_PATH, 'correct_preds')
    data = datasets.ImageFolder(data_dir, transform=transforms.Compose(
        [
            transforms.ToTensor(), # no need for the centercrop as it is at the cor
            transforms.Normalize(
                [Constants.DATA_MEAN, Constants.DATA_MEAN, Constants.DATA_MEAN], 
                [Constants.DATA_STD,Constants.DATA_STD, Constants.DATA_STD])
        ]
    ))

    # for each image, it has a folder that store all the cam heatmaps
    dataloader = DataLoader(data, batch_size=len(data)) # TODO: check image 18
    x, _ = next(iter(dataloader))
    cam = switch_cam(args.cam, model_wrapper.model, model_target_layer) 

    print('--------- Forward Passing {}'.format(args.cam))
    input_x = x
    if args.noiseSmooth:
        grayscale_cam = torch.zeros((x.shape[0], x.shape[-1], x.shape[-1]), dtype=Constants.DTYPE)
        for t in range(args.iterations):
            print('CAM Smoothing Iteration: {}'.format(t))
            input_x = add_noise(x, args.std)
            grayscale_cam += cam(input_tensor=input_x, targets=None)
        grayscale_cam /= args.iterations
    else:
        grayscale_cam = cam(input_tensor=input_x, targets=None)
    
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
        img_name = '{}-{}layers'.format(args.cam, len(model_target_layer))
        masked_img.save(os.path.join(dest, img_name+'.jpg'))