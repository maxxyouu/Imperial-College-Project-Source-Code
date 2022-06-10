import torch
import timm
import torch.nn as nn
import Constants

class Baseline_Model:
    """
    Baseline model directly obtain from the pytorch library
    https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    """
    def __init__(self, pretrain=False, model_name='resnet18') -> None:
        # self.model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=pretrain)
        self.model = timm.create_model(model_name, pretrained=pretrain)
        # modify the model that suit our task, ie: the output layer and etc

    def _custom_classifier(self):
        """change the classifier part of the default that match our part
        """
        assert(self.model is not None)

    def model_size(self):
        """ return number of parameters in the model
        reference: code refer to the coursework in DL
        """
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return params

    def view_model(self):
        print(self.model.parameters)
    
    def load_learned_weights(self, weight_pickle='model.pt'):
        self.model.load_state_dict(torch.load(weight_pickle, map_location=Constants.DEVICE))
        self.model.to(Constants.DEVICE)

class Pytorch_default_resNet(Baseline_Model):
    """default resnet model from pytorch

    Args:
        Baseline_Model (_type_): _description_
    """
    def __init__(self, dtype=Constants.DTYPE, device=Constants.DEVICE, num_classes=2, pretrain=False, model_name='resnet18') -> None:
        super().__init__(pretrain, model_name)

        # modify the model to match our dataset with two class only
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes, device=device, dtype=dtype)

class Pytorch_default_vgg(Baseline_Model):
    def __init__(self, dtype=Constants.DTYPE, device=Constants.DEVICE, num_classes=2, pretrain=False, model_name='vgg11_bn') -> None:
        super().__init__(pretrain, model_name)
        # NOTE: the implementation is different for both timm and pytorch 
        # modify the model to match our dataset with two class only

        # for timmm
        # self.model.head.fc = 

        # for pytorch
        self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, num_classes, device=device, dtype=dtype)

class Pytorch_default_skres(Baseline_Model):
    """https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/sknet.py

    Args:
        Baseline_Model (_type_): _description_
    """

    def __init__(self, dtype=Constants.DTYPE, device=Constants.DEVICE, num_classes=2, pretrain=False, model_name='skresnet18') -> None:
        super().__init__(pretrain, model_name)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes, device=device, dtype=dtype)
        
class Pytorch_default_resnext(Baseline_Model):
    """https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/sknet.py

    Args:
        Baseline_Model (_type_): _description_
    """

    def __init__(self, dtype=Constants.DTYPE, device=Constants.DEVICE, num_classes=2, pretrain=False, model_name='resnext50_32x4d') -> None:
        super().__init__(pretrain, model_name)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes, device=device, dtype=dtype)


class Pytorch_default_skresnext(Baseline_Model):
    """https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/sknet.py

    Args:
        Baseline_Model (_type_): _description_
    """

    def __init__(self, dtype=Constants.DTYPE, device=Constants.DEVICE, num_classes=2, pretrain=False, model_name='skresnet18') -> None:
        super().__init__(pretrain, model_name)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes, device=device, dtype=dtype)

if __name__ == '__main__':
    # print(torch. __version__)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)

    net = Pytorch_default_skres(dtype=Constants.DTYPE, device=Constants.DEVICE)
    # net.view_model()
    print("hello")