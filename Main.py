from progressbar import DataTransferBar
import torch
from torch import nn
import torchvision
import torch.nn.functional as F
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

# local file import
from CLEImageDataset import CLEImageDataset
from BaselineModel import Pytorch_default_resNet
from Helper import extract_args, main_executation, data_transformations, pytorch_dataset
import Constants
from Helper import denorm

# set the seed for reproducibility
rng_seed = 99
torch.manual_seed(rng_seed)
print('Device being used: {}'.format(Constants.DEVICE))

# main class responsible for training
class Main:
    def __init__(self, args):
        def _create_model_name(args):
            name = args['model_name']
            if args['augNoise']:
                name += '_noise'
            if args['pretrain']:
                name += '_pretrain'
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
    
    def _extract_correct_preds_and_save(self, preds, y, features):
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

        # create directory the model's subdirectory if not exists
        dest = os.path.join(Constants.STORAGE_PATH, 'correct_preds')
        if not os.path.exists(dest):
            os.mkdir(dest)
        dest = os.path.join(dest, self.model_name)
        if not os.path.exists(dest):
            os.mkdir(dest)

        for i, (_, label) in enumerate(zip(correct_pred_samples, correct_classified_labels)):
            torchvision.utils.save_image(correct_pred_samples[i, :, :, :], os.path.join(dest, '{}-label{}.jpg'.format(i, label)))

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
            for x, y in loader:

                x = x.to(device=Constants.DEVICE, dtype=Constants.DTYPE)  # move to device
                y = y.to(device=Constants.DEVICE, dtype=torch.long)

                scores = self.model_wrapper.model(x)
                _, preds = scores.max(1)

                # record the loss
                losses.append(self.loss(scores, y))

                if store_sample:
                    print('Saving correctly classified images')
                    self._extract_correct_preds_and_save(preds, y, x)

                num_correct += (preds == y).sum()
                num_samples += preds.size(0)

        acc = float(num_correct) / num_samples

        if _print:
            print('Accuracy: {}'.format(acc))
        return float(acc), sum(losses)/len(losses)
        
    def train(self, model_weights_des=Constants.SAVED_MODEL_PATH):
        self.model_wrapper.model = self.model_wrapper.model.to(device=Constants.DEVICE)  # move the model parameters to CPU/GPU

        patience, optimal_val_loss = 5, np.inf

        for e in range(self.epochs):
            for t, (x, y) in enumerate(self.loader_train):
                self.optimizer.zero_grad()

                # add gaussian noise
                if self.addNoise:
                    noise = torch.zeros(x.shape, dtype=Constants.DTYPE) + (0.1**0.5)*torch.randn(x.shape)
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


if __name__ == '__main__':

    # extract argument from users
    args = extract_args()

    train_transforms, test_transforms = data_transformations()

    data_dict = pytorch_dataset(args.batchSize, train_transforms, test_transforms)
    train, train_dataloader = data_dict['train']
    val, val_dataloader = data_dict['val']
    test, test_dataloader = data_dict['test']

    model_wrapper = Pytorch_default_resNet(device=Constants.DEVICE, dtype=Constants.DTYPE, model_name=args.model, pretrain=args.pretrain)
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
        'pretrain': args.pretrain
    }

    main = Main(params)

    # decide the execution mode
    main_executation(args.train, main)
