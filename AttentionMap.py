from ast import Constant
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

# local imports
import Constants
from Helper import denorm, switch_cam, extract_attention_cam_args, get_trained_model, find_mutual_correct_images
        
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

# TODO: the following only valid for resnet 50 and resnext 50
def target_layers(model, layer_nums):
    result = []
    for i in layer_nums:
        result.append(getattr(model, 'layer'+'{}'.format(i))[-1])
    return result


if __name__ == '__main__':

    # get all mutual correct predictions
    # find_mutual_correct_images(os.path.join(Constants.STORAGE_PATH, 'mutual_corrects'))

    # need to manually modify the smooth parameters
    args = extract_attention_cam_args()

    # print statement to verify the boolean arguments
    print('Noise Smooth Arg: {}'.format(args.noiseSmooth))
    print('Positive Target Arg: {}'.format(args.positiveTarget))

    model_wrapper = get_trained_model(args.model)
    model_target_layer = target_layers(model_wrapper.model, args.layers)

    model_dir_name = define_model_dir_path(args)
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
    dataloader = DataLoader(data, batch_size=args.batchSize, sampler=sequentialSampler) # TODO: check image 18
    image_order_book, img_index = data.imgs, 0
    cam = switch_cam(args.cam, model_wrapper.model, model_target_layer)
    for x, y in dataloader:
        # NOTE: make sure i able index to the correct index
        print('--------- Forward Passing {}'.format(args.cam))
        x = x.to(device=Constants.DEVICE, dtype=Constants.DTYPE)
        input_x = x
        if args.noiseSmooth:
            grayscale_cam = torch.zeros((x.shape[0], x.shape[-1], x.shape[-1]),dtype=Constants.DTYPE)
            for t in range(args.iterations):
                # print('CAM Smoothing Iteration: {}'.format(t))
                input_x = add_noise(x, args.std)
                grayscale_cam += cam(input_tensor=input_x, targets=None)
            grayscale_cam /= args.iterations
        else:
            grayscale_cam = cam(input_tensor=input_x, targets=None)
        
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
            img = np.swapaxes(img, 0, 2)
            img = np.swapaxes(img, 0, 1)

            # save the overlayed-attention map with the cam name as a tag
            attention_map = show_cam_on_image(img, grayscale_cam[i, :], use_rgb=True)
            masked_img = Image.fromarray(attention_map, 'RGB')
            cam_name = '{}-{}layers'.format(args.cam, len(model_target_layer))
            masked_img.save(os.path.join(dest, cam_name+'.jpg'))

            # update the sequential index for next iterations
            img_index += 1