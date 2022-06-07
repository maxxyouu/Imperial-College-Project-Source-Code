
import torch
from torch import nn
import torch.nn.functional as F
import os
import torch.optim as optim

# local file import
from CLEImageDataset import CLEImageDataset
from BaselineModel import Pytorch_default_resNet
from torch.utils.data import DataLoader
from torchvision import transforms


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
        
        # logit
        # self.softmax = nn.Softmax(dim=1) # TODO: check the dimension

    def check_accuracy(self, loader):
        # function for test accuracy on validation and test set
        
        num_correct = 0
        num_samples = 0
        self.model_wrapper.model.eval()  # set model to evaluation mode
        with torch.no_grad():
            for x, y in loader:

                x = x.to(device=DEVICE, dtype=DTYPE)  # move to device
                y = y.to(device=DEVICE, dtype=torch.long)

                scores = self.model_wrapper.model(x)
                _, preds = scores.max(1)
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)

            acc = float(num_correct) / num_samples
            # print('Got %d / %d correct of val set (%.2f)' % (num_correct, num_samples, 100 * acc))

            return float(acc)
        
    def train(self, epochs, print_every=5, model_weights_des='../'):
        self.model_wrapper.model = self.model_wrapper.model.to(device=DEVICE)  # move the model parameters to CPU/GPU
        
        opt_val_acc = 0

        for e in range(epochs):
            for t, (x, y) in enumerate(self.loader_train):
                self.optimizer.zero_grad()

                self.model_wrapper.model.train()  # put model to training mode
                x = x.to(device=DEVICE, dtype=DTYPE)  # move to device, e.g. GPU
                y = y.to(device=DEVICE, dtype=torch.long)

                unnormalized_score = self.model_wrapper.model(x) # unnormalized
                # loss = F.cross_entropy(scores, y) # TODO: make sure it is appropriate
                loss = self.loss(unnormalized_score, y) # TODO: make sure it is appropriate

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
                    torch.save(self.model_wrapper.model.state_dict(), model_dest)

def mu_std(data_loader):
    count = 0
    mean, var = 0, 0
    for batch_features, _ in data_loader:
        # Rearrange batch to be the shape of [B, C, W * H]
        batch_features = batch_features.view(batch_features.size(0), batch_features.size(1), -1)
        # Update total number of images
        count += batch_features.size(0)
        # Compute mean and std here
        mean += batch_features.mean(2).sum(0) 
        var += batch_features.var(2).sum(0)

    mean /= count
    var /= count
    std = torch.sqrt(var)
    return mean, std


if __name__ == '__main__':
    BATCH_SIZE = 3
    transforms = transforms.Compose([
        transforms.ToTensor(), 
        # transforms.Grayscale(1),
        transforms.CenterCrop((336, 350)),
        transforms.Normalize([0.1496,0.1496,0.1496], [0.1960,0.1960,0.1960])
    ])

    train = CLEImageDataset('../train', annotations_file='../train_annotations.csv', transform=transforms)
    val = CLEImageDataset('../val', annotations_file='../val_annotations.csv', transform=transforms)
    test = CLEImageDataset('../test', annotations_file='../test_annotations.csv', transform=transforms)

    train_dataloader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)

    resnet18 = Pytorch_default_resNet(device=DEVICE, dtype=DTYPE, model_name='resnet18')

    params = {
        'train_data': train,
        'loader_train': train_dataloader,
        'val_data': val,
        'loader_val': val_dataloader,
        'test_data': test,
        'loader_test': test_dataloader,
        'model': resnet18,
        'optimizer': optim.Adamax(resnet18.model.parameters(), lr=0.001, weight_decay=1e-8),
        'loss': nn.CrossEntropyLoss()
    }
    
    main = Main(params)
    main.train(5)

    ## find the mean and variance
    # all_data = CLEImageDataset('../cleanDistilledFrames', transform=transforms.Compose([transforms.ToTensor()]))
    # all_data_loader = DataLoader(all_data, batch_size=100, shuffle=True)
    # mus, stds = mu_std(all_data_loader)
    # print(mus)
    # print(stds)

