from torchvision import transforms, datasets
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

# local imports
from BaselineModel import Pytorch_default_resNet
from CLEImageDataset import CLEImageDataset
import Constants

if __name__ == '__main__':
    model_name = 'resnet18'
    resnet18 = Pytorch_default_resNet(model_name=model_name)
    resnet18.load_learned_weights('./trained_models/{}.pt'.format(model_name))
    resnet18_target_layer = resnet18.model.layer4[-1]
    cam = GradCAM(model=resnet18.model, target_layers=[resnet18_target_layer])

    data = datasets.ImageFolder('./correct_preds', transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                [Constants.DATA_MEAN, Constants.DATA_MEAN, Constants.DATA_MEAN], 
                [Constants.DATA_STD,Constants.DATA_STD, Constants.DATA_STD])
        ]
    ))
    dataloader = DataLoader(data, batch_size=2, shuffle=True)
    # for x, _ in dataloader:
    #     grayscale_cam = cam(input_tensor=x, targets=None)
    #     print('hello')
    
    x, _ = next(iter(dataloader))

    # tensorToImg = transforms.ToPILImage()
    # img = tensorToImg(x[0, :])
    grayscale_cam = cam(input_tensor=x, targets=None)[0, :]
    x.mul_(Constants.DATA_STD).add_(Constants.DATA_MEAN)
    img = x[0, :].cpu().detach().numpy()
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 0, 1)
    visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)

    camed_image = Image.fromarray(visualization, 'RGB')
    camed_image.show()
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")

