import torch
import argparse

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


def extract_args():
    my_parser = argparse.ArgumentParser(description='')

    # Add the arguments
    my_parser.add_argument('--model',
                            type=str, default='resnet18',
                            help='model to be used for training / testing')
    my_parser.add_argument('--pretrain',
                            type=bool, default=False,
                            help='whether to use a pretrained model')
    my_parser.add_argument('--batchSize',
                            type=int, default=256,
                            help='batch size to be used for training / testing')             
    my_parser.add_argument('--epochs',
                            type=int, default=30,
                            help='training epochs')   
    my_parser.add_argument('--earlyStoppingPatience',
                            type=int, default=5,
                            help='early stopping patience to terminate the training process')   
    my_parser.add_argument('--learningRate',
                            type=float, default=0.001,
                            help='learning rate for training')   

    # Execute the parse_args() method
    args = my_parser.parse_args()                                              

    return args