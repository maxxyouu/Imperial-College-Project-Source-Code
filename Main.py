from copy import deepcopy
from collections import OrderedDict
import torch
from torch import nn
import torchvision
import os
import torch.optim as optim
import numpy as np
from BaselineModel import Pytorch_default_skresnext
from resnet_big import *
import timm
# local file import
from Helper import main_executation, data_transformations, pytorch_dataset, switch_model
import Constants
from Helper import denorm, get_trained_model
import argparse

my_parser = argparse.ArgumentParser(description='')

# Add the arguments
my_parser.add_argument('--model',
                        type=str, default='skresnext50_32x4d',
                        help='model to be used for training / testing')
my_parser.add_argument('--batchSize',
                        type=int, default=256,
                        help='batch size to be used for training / testing')             
my_parser.add_argument('--epochs',
                        type=int, default=100,
                        help='training epochs')   
my_parser.add_argument('--earlyStoppingPatience',
                        type=int, default=10,
                        help='early stopping patience to terminate the training process')   
my_parser.add_argument('--learningRate',
                        type=float, default=0.001,
                        help='learning rate for training') 
my_parser.add_argument('--pretrain',
                        type=bool, action=argparse.BooleanOptionalAction,
                        help='whether to use a pretrained model')
my_parser.add_argument('--augNoise',
                        type=bool, action=argparse.BooleanOptionalAction,
                        help='add noise during traning')   
my_parser.add_argument('--train',
                        type=bool, action=argparse.BooleanOptionalAction,
                        help='whether execute the script in training or eval mode')   
my_parser.add_argument('--chkPointName',
                        type=str, default='', # example: ckpt_epoch_500
                        help='the check point name')  
my_parser.add_argument('--simClr',
                        type=bool, action=argparse.BooleanOptionalAction, # example: ckpt_epoch_500
                        help='for simclr task') 
my_parser.add_argument('--supCon',
                        type=bool, action=argparse.BooleanOptionalAction, # example: ckpt_epoch_500
                        help='for supCon task') 
my_parser.add_argument('--feat_dim',
                        type=int, default=64, # example: ckpt_epoch_500
                        help='feature dimension of the loaded clr model') 
my_parser.add_argument('--head_type',
                        type=str, default='mlp', # example: ckpt_epoch_500
                        help='head type of the clr model') 
my_parser.add_argument('--pickel_initial',
                        type=str, default='', # example: ckpt_epoch_500
                        help='intial name for the saved  mdel') 
my_parser.add_argument('--folderName',
                        type=str, default=None, # example: ckpt_epoch_500
                        help='eg: SupCon_path_skresnext50_32x4d_lr_0.05_decay_0.0001_bsz_128_temp_0.07_trial_0_64_mlp_cosine') 
my_parser.add_argument('--headWidth',
                        type=int, default=1, # example: ckpt_epoch_500
                        help='width of the projection head of the classifier') 
my_parser.add_argument('--layerDropout',
                        type=bool, action=argparse.BooleanOptionalAction, 
                        help='add one dropout layer to each stage')                   
my_parser.add_argument('--withDropout',
                        type=bool, action=argparse.BooleanOptionalAction, 
                        help='with layerDropout being true, use this')         

# Execute the parse_args() method
args = my_parser.parse_args()
args.pickel_initial += '_headWidth{}'.format(args.headWidth)
args.pickel_initial += '_withLayerDropout' if args.withDropout else ''
if args.pretrain is None:
    args.pretrain = True

# set the seed for reproducibility
rng_seed = 99
torch.manual_seed(rng_seed)
print('Device being used: {}'.format(Constants.DEVICE))

# main class responsible for training
class Main:
    def __init__(self, args):
        def _create_model_name(args):
            name = args['model_name'] + args['pickel_initial']
            if args['augNoise']:
                name += '_noise'
            if args['pretrain']:
                name += '_pretrain'
            if args['simClr']:
                name += '_simclr'
            elif args['supCon']:
                name += '_supCon'
            if args['chkPointName']:
                name = name + '_' + args['chkPointName'][:-4]
            return name

        # a pytorch model
        self.model_wrapper = args['model']
        self.model_name = _create_model_name(args)

        # store the dataset
        self.training_data = args['train_data']
        self.loader_train = args['loader_train']
        self.val_data = args['val_data']
        self.loader_val = args['loader_val']
        self.test_data = args['test_data']
        self.loader_test = args['loader_test']

        # define a optimizer
        self.optimizer = args['optimizer']

        # define a loss function
        self.loss = args['loss'] #nn.CrossEntropyLoss()
        
        # for early stopping
        self.earlyStopping_patience = args['patience']
        self.epochs = args['epochs']

        self.addNoise = args['augNoise']
    
    def _extract_correct_preds_and_save(self, preds, y, features, names):
        """store the correctedly classified sample to the corresponding folder

        Args:
            preds (_type_): prediction from model
            y (_type_): labels
            features (_type_): 4d tensor
        """
        # denorm the features
        features = denorm(features)

        correct_classifications = (preds == y)
        
        # get the corresponding images from the batch
        correct_pred_samples = features[correct_classifications, :, :, :]
        # get the corresponding label
        correct_classified_labels = y[correct_classifications]
        names = np.array(names)
        names = names[correct_classifications.cpu()]

        # create directory the model's subdirectory if not exists
        dest_0 = os.path.join(Constants.STORAGE_PATH, 'correct_preds', self.model_name, '0')
        dest_1 = os.path.join(Constants.STORAGE_PATH, 'correct_preds', self.model_name, '1')
        if not os.path.exists(dest_0):
            os.makedirs(dest_0)
        if not os.path.exists(dest_1):
            os.makedirs(dest_1)

        for i, (_, label, img_name) in enumerate(zip(correct_pred_samples, correct_classified_labels, names)):
            dest = dest_1 if label.item() == 1 else dest_0
            # sanity check
            if label.item() == 1 and 'meningioma' not in img_name:
                print('label={}-{}'.format(label.item(), img_name))
                continue
            elif label.item() == 0 and 'GBM' not in img_name:
                print('label={}-{}'.format(label.item(), img_name))
                continue
            torchvision.utils.save_image(correct_pred_samples[i, :, :, :], os.path.join(dest, img_name))

    def check_accuracy(self, loader, best_model=False, store_sample=False, _print=False):
        # function for test accuracy on validation and test set
        
        if best_model:
            trained_model_loc = os.path.join(Constants.SAVED_MODEL_PATH, '{}.pt'.format(self.model_name))
            self.model_wrapper.load_learned_weights(trained_model_loc)
            print('Finished Loading the Best Model for {}'.format(self.model_name))

        num_correct = 0
        num_samples = 0
        acc, losses = 0, []

        self.model_wrapper.model.eval()  # set model to evaluation mode
        with torch.no_grad():
            for x, y, names in loader:

                x = x.to(device=Constants.DEVICE, dtype=Constants.DTYPE)  # move to device
                y = y.to(device=Constants.DEVICE, dtype=torch.long)

                scores = self.model_wrapper.model(x)
                _, preds = scores.max(1)

                # record the loss
                losses.append(self.loss(scores, y))

                if store_sample:
                    print('Saving correctly classified images')
                    self._extract_correct_preds_and_save(preds, y, x, names)

                num_correct += (preds == y).sum()
                num_samples += preds.size(0)

        acc = float(num_correct) / num_samples

        if _print:
            print('Accuracy: {}'.format(acc))
        return float(acc), sum(losses)/len(losses)
        
    def train(self, model_weights_des=Constants.SAVED_MODEL_PATH):
        self.model_wrapper.model = self.model_wrapper.model.to(device=Constants.DEVICE)  # move the model parameters to CPU/GPU

        patience, optimal_val_loss = self.earlyStopping_patience, np.inf

        for e in range(self.epochs):
            for t, (x, y, name) in enumerate(self.loader_train):
                self.optimizer.zero_grad()

                # add gaussian noise
                if self.addNoise:
                    # refer to smoothgrad paper
                    noise = torch.zeros(x.shape, dtype=Constants.DTYPE) + torch.randn(x.shape) / (torch.max(x) - torch.min(x))
                    x += noise

                self.model_wrapper.model.train()  # put model to training mode
                x = x.to(device=Constants.DEVICE, dtype=Constants.DTYPE)  # move to device, e.g. GPU
                y = y.to(device=Constants.DEVICE, dtype=torch.long)

                unnormalized_score = self.model_wrapper.model(x) # unnormalized
                loss = self.loss(unnormalized_score, y) # TODO: make sure it is appropriate

                # Update the parameters of the model using the gradients
                loss.backward()
                self.optimizer.step()

                # TODO: might need to save the training loss for ploting using t in the x axis

            # evaluate the validation dataset after every epoch
            val_acc, val_loss = self.check_accuracy(self.loader_val)
            print('Epoch: {}, Validation Loss {}, val_acc {}'.format(e, val_loss, val_acc))

            if val_loss < optimal_val_loss:
                print('Saving model')
                # save the model to destination
                model_dest = os.path.join(model_weights_des, '{}.pt'.format(self.model_name))
                torch.save(self.model_wrapper.model.state_dict(), model_dest)

                # udpate the tracking parameters
                optimal_val_loss = val_loss
                patience = self.earlyStopping_patience
            else:
                patience -= 1

            # stop training when epoch acc no longer improve for consecutive {patience} epochs
            if patience <= 0:
                break

def compatible_weights(clr_weights):
    result = []
    for key, value in clr_weights.items():
        key_copy = key
        if 'encoder.' in key:
            key_copy = key.replace('encoder.', '')
        elif 'head.' in key:
            be_replaced = key[:len('head')] # + 1 for the head.0, head.4, and etc to be fc.0, fc.4 and etc
            key_copy = key.replace(be_replaced, 'fc')
        result.append((key_copy, value))
    return OrderedDict(result)

# print statement to verify the bool arguments
print('Pretrain Arg: {}'.format(args.pretrain))
print('augNoise Arg: {}'.format(args.augNoise))
print('train Arg: {}'.format(args.train))
print('simclr: {}'.format(args.simClr))
print('supCon: {}'.format(args.supCon))
print('chkPointName: {}'.format(args.chkPointName))
print('model path: {}'.format(args.folderName))
print('layer dropout {}'.format(args.layerDropout))
print('dropout rate {}'.format(args.dropoutRate))

train_transforms, test_transforms = data_transformations()

data_dict = pytorch_dataset(args.batchSize, train_transforms, test_transforms)
train, train_dataloader = data_dict['train']
val, val_dataloader = data_dict['val']
test, test_dataloader = data_dict['test']

if args.simClr or args.supCon: # TODO
    if args.simClr:
        assert('Sim' in args.folderName)
        clr_weight_path = Constants.SIMCLR_MODEL_PATH.format(args.folderName)
    elif args.supCon:
        assert('Sup' in args.folderName)
        clr_weight_path = Constants.SUPCON_MODEL_PATH.format(args.folderName)
    print(clr_weight_path)
    assert(args.chkPointName is not None and args.model == 'skresnext50_32x4d' and args.folderName is not None)
    model_weights_loc = os.path.join(Constants.SAVED_MODEL_PATH, clr_weight_path, args.chkPointName)
    # load the skresnext model and replace the head with a appropriate one
    clr_model = SimClrSkResneXt(name=args.model, head=args.head_type, feat_dim=args.feat_dim) # with pretrained weight in the feature extractor
    model_wrapper = Pytorch_default_skresnext()
    model_wrapper.model.fc = deepcopy(clr_model.head) # NOTE: random but to be replaced with trained weights
    saved_model = torch.load(model_weights_loc, map_location=Constants.DEVICE) # check util.py

    pickel = compatible_weights(saved_model['model'])
    model_wrapper.model.fc = nn.Sequential(
            nn.Linear(2048, 2048 // 2),
            nn.ReLU(inplace=True),
            nn.Linear(2048 // 2, 2048 // 4),
            nn.ReLU(inplace=True),
            nn.Linear(2048 // 4, 256)
    )
    model_wrapper.model.load_state_dict(pickel) # get the state dict
    model_wrapper.model.to(Constants.DEVICE)
    
    # retraining start from the middle of the training head 
    # model_wrapper.model.fc[0] = nn.Linear(model_wrapper.model.fc[0].in_features, model_wrapper.model.fc[0].out_features)
    # model_wrapper.model.fc[2] = nn.Linear(model_wrapper.model.fc[2].in_features, model_wrapper.model.fc[2].out_features)
    # model_wrapper.model.fc[4] = nn.Linear(model_wrapper.model.fc[4].in_features, 2)

    # experiment show that small width projection head works better visually
    model_wrapper.model.fc = nn.Linear(model_wrapper.model.fc[0].in_features, 2)
elif args.train:
    model_wrapper = switch_model(args.model, args.pretrain, headWidth=args.headWidth)
else:
    model_wrapper = get_trained_model(args.model)

params = {
    'train_data': train,
    'loader_train': train_dataloader,
    'val_data': val,
    'loader_val': val_dataloader,
    'test_data': test,
    'loader_test': test_dataloader,
    'model': model_wrapper,
    'optimizer': optim.Adamax(model_wrapper.model.parameters(), lr=args.learningRate, weight_decay=1e-8),
    'loss': nn.CrossEntropyLoss(),
    'model_name': args.model,
    'patience': args.earlyStoppingPatience,
    'epochs': args.epochs,
    'augNoise': args.augNoise,
    'pretrain': args.pretrain,
    'simClr': args.simClr,
    'supCon': args.supCon,
    'chkPointName': args.chkPointName,
    'pickel_initial': args.pickel_initial
}

main = Main(params)

# print the number of parameters
print('Number of parameters: {}'.format(model_wrapper.model_size()))

# decide the execution mode
main_executation(main, args.train)
