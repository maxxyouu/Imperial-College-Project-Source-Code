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
from BaselineModel import Pytorch_default_resNet
import Constants

class CAM_Generator:
    def __init__(self, model_name, data) -> None:
        pass

if __name__ == '__main__':

    model_name = 'resnet18'
    resnet18 = Pytorch_default_resNet(model_name=model_name)
    resnet18.load_learned_weights('./trained_models/{}.pt'.format(model_name))
    resnet18_target_layer = resnet18.model.layer4[-1]

    data = datasets.ImageFolder('./correct_preds', transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                [Constants.DATA_MEAN, Constants.DATA_MEAN, Constants.DATA_MEAN], 
                [Constants.DATA_STD,Constants.DATA_STD, Constants.DATA_STD])
        ]
    ))

    # for each image, it has a folder that store all the cam heatmaps
    dataloader = DataLoader(data, batch_size=len(data))
    x, _ = next(iter(dataloader))

    cams = ['xgradcam'] # 'scorecam', 'ablationcam', 'xgradcam', 'eigencam',
    for cam_name in cams:
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
                dest = os.path.join(Constants.STORAGE_PATH, 'heatmaps', model_name, 'image-{}'.format(i))
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