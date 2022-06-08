
import torch
from torch import nn
import torch.nn.functional as F
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

# local file import
from CLEImageDataset import CLEImageDataset
from BaselineModel import Pytorch_default_resNet
from Helper import extract_args

# WORK_ENV = 'COLAB'
WORK_ENV = 'LOCAL'
DATA_PARENT_PATH = '../'
if WORK_ENV == 'COLAB':
    DATA_PARENT_PATH = '/content/drive/MyDrive/CLEdata/'

import progressbar

# set the seed for reproducibility
rng_seed = 99
torch.manual_seed(rng_seed)
USE_GPU = True
# training device
DTYPE = torch.float32
DEVICE = torch.device('cpu')
if USE_GPU and torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
print('Device being used: {}'.format(DEVICE))

# main class responsible for training
class Main:
    def __init__(self, args):

        # a pytorch model
        self.model_wrapper = args['model']
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
        
        # for early stopping
        self.earlyStopping_patience = args['patience']
        self.epochs = args['epochs']

    def check_accuracy(self, loader):
        # function for test accuracy on validation and test set
        
        num_correct = 0
        num_samples = 0
        self.model_wrapper.model.eval()  # set model to evaluation mode

        with torch.no_grad():
            for x, y in progressbar.progressbar(self.loader_val):

                x = x.to(device=DEVICE, dtype=DTYPE)  # move to device
                y = y.to(device=DEVICE, dtype=torch.long)

                scores = self.model_wrapper.model(x)
                _, preds = scores.max(1)
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)
            acc = float(num_correct) / num_samples
            
            return float(acc)
        
    def train(self, print_every=5, model_weights_des='../'):
        self.model_wrapper.model = self.model_wrapper.model.to(device=DEVICE)  # move the model parameters to CPU/GPU

        opt_val_acc = 0
        patience, optimal_epoch_acc = 5, 0

        for e in range(self.epochs):
            epoch_val_acc  = []
            for t, (x, y) in enumerate(progressbar.progressbar(self.loader_train)):
                self.optimizer.zero_grad()

                # add gaussian noise
                noise = torch.zeros(x.shape, dtype=DTYPE) + (0.1**0.5)*torch.randn(x.shape)
                x += noise

                self.model_wrapper.model.train()  # put model to training mode
                x = x.to(device=DEVICE, dtype=DTYPE)  # move to device, e.g. GPU
                y = y.to(device=DEVICE, dtype=torch.long)

                unnormalized_score = self.model_wrapper.model(x) # unnormalized
                loss = self.loss(unnormalized_score, y) # TODO: make sure it is appropriate

                # Zero out all of the gradients for the variables which the optimizer
                # will update.

                # Update the parameters of the model using the gradients
                loss.backward()
                self.optimizer.step()
                
                # log training process
                val_acc = self.check_accuracy(self.loader_val)
                epoch_val_acc.append(val_acc)
                if t % print_every == 0:
                    print('Epoch: {}, Iteration {}, loss {}, val_acc {}'.format(e, t, loss.item(), val_acc))

                # save the model if it is currently the optimal
                if val_acc > opt_val_acc:
                    print('Saving model')
                    # update the current optimal validation acc
                    opt_val_acc = val_acc

                    # save the model to destination
                    model_dest = os.path.join(model_weights_des, '{}.pt'.format(self.model_name))
                    torch.save(self.model_wrapper.model.state_dict(), model_dest)

            # average epoch acc
            epoch_acc = sum(epoch_val_acc)/len(epoch_val_acc)
            print('Epoch {} validation acc: {}'.format(e, epoch_acc))

            if optimal_epoch_acc < epoch_acc:
                 epoch_acc = optimal_epoch_acc
                 patience = self.earlyStopping_patience
            else:
                patience -= 1

            # stop training when epoch acc no longer improve for consecutive {patience} epochs
            if patience <= 0:
                break


if __name__ == '__main__':

    # extract argument from users
    args = extract_args()

    transforms = transforms.Compose([
        transforms.ToTensor(), 
        # transforms.Grayscale(1),
        transforms.CenterCrop((336, 350)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((0, 270)),
        transforms.Normalize([0.1496,0.1496,0.1496], [0.1960,0.1960,0.1960])
    ])

    if WORK_ENV == 'COLAB':
        train_datapath = '{}train'.format(DATA_PARENT_PATH)
        val_datapath = '{}val'.format(DATA_PARENT_PATH)
        test_datapath = '{}test'.format(DATA_PARENT_PATH)

        train_annotationPath = '{}train_annotations.csv'.format(DATA_PARENT_PATH)
        val_annotationPath = '{}val_annotations.csv'.format(DATA_PARENT_PATH)
        test_annotationPath = '{}test_annotations.csv'.format(DATA_PARENT_PATH)

    else: # local
        train_datapath = '{}train'.format(DATA_PARENT_PATH)
        val_datapath = '{}val'.format(DATA_PARENT_PATH)
        test_datapath = '{}test'.format(DATA_PARENT_PATH)

        train_annotationPath = '{}train_annotations.csv'.format(DATA_PARENT_PATH)
        val_annotationPath = '{}val_annotations.csv'.format(DATA_PARENT_PATH)
        test_annotationPath = '{}test_annotations.csv'.format(DATA_PARENT_PATH)

    train = CLEImageDataset(train_datapath, annotations_file=train_annotationPath, transform=transforms)
    val = CLEImageDataset(val_datapath, annotations_file=val_annotationPath, transform=transforms)
    test = CLEImageDataset(test_datapath, annotations_file=test_annotationPath, transform=transforms)

    train_dataloader = DataLoader(train, batch_size=args.batchSize, shuffle=True)
    val_dataloader = DataLoader(val, batch_size=args.batchSize, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=args.batchSize, shuffle=True)

    model_wrapper = Pytorch_default_resNet(device=DEVICE, dtype=DTYPE, model_name=args.model, pretrain=args.pretrain)
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
        'epochs': args.epochs
    }
    
    main = Main(params)
    print('Training Started')
    main.train(20)

    ## find the mean and variance
    # all_data = CLEImageDataset('../cleanDistilledFrames', transform=transforms.Compose([transforms.ToTensor()]))
    # all_data_loader = DataLoader(all_data, batch_size=100, shuffle=True)
    # mus, stds = mu_std(all_data_loader)
    # print(mus)
    # print(stds)

