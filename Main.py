
import torch
from torch import nn
import torch.nn.functional as F
import os

# local file import
from CLEImageDataset import CLEImageDataset
from BaselineModel import Pytorch_default_resNet

# set the seed for reproducibility
rng_seed = 99
torch.manual_seed(rng_seed)
USE_GPU = True

# main class responsible for training
class Main:
    def __init__(self, **args):

        # a pytorch model
        self.model = args['model']
        self.model_name = args['model_name']

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

        # training device
        self.dtype = torch.float32
        self.device = torch.device('cpu')
        if USE_GPU and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        print('Device being used: {}'.format(self.device))

    def check_accuracy(self, loader):
        # function for test accuracy on validation and test set
        
        num_correct = 0
        num_samples = 0
        self.model.eval()  # set model to evaluation mode
        with torch.no_grad():
            for x, y in loader:

                x = x.to(device=self.device, dtype=self.dtype)  # move to device
                y = y.to(device=self.device, dtype=torch.long)

                scores = self.model(x)
                _, preds = scores.max(1)
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)

            acc = float(num_correct) / num_samples
            # print('Got %d / %d correct of val set (%.2f)' % (num_correct, num_samples, 100 * acc))

            return float(acc)
        
    def train(self, epochs, print_every=5, model_weights_des='../'):
        self.model = self.model.to(device=self.device)  # move the model parameters to CPU/GPU
        
        opt_val_acc = 0

        for e in range(epochs):
            for t, (x, y) in enumerate(self.loader_train):
                self.optimizer.zero_grad()

                self.model.train()  # put model to training mode
                x = x.to(device=self.device, dtype=self.dtype)  # move to device, e.g. GPU
                y = y.to(device=self.device, dtype=torch.long)

                scores = self.model(x)
                # loss = F.cross_entropy(scores, y) # TODO: make sure it is appropriate
                loss = self.loss(scores, y) # TODO: make sure it is appropriate

                # Zero out all of the gradients for the variables which the optimizer
                # will update.

                # Update the parameters of the model using the gradients
                loss.backward()
                self.optimizer.step()
                
                # log training process
                val_acc = self.check_accuracy(self.loader_val)
                if t % print_every == 0:
                    print('Epoch: {}, Iteration {}, loss {}, val_acc {}'.format(e, t, loss.item(), val_acc))

                # save the model if it is currently the optimal
                if val_acc > opt_val_acc:
                    # update the current optimal validation acc
                    opt_val_acc = val_acc

                    # save the model to destination
                    model_dest = os.path.join(model_weights_des, '{}.pt'.format(self.model_name))
                    torch.save(self.model.state_dict(), model_dest)



if __name__ == '__main__':
    pass