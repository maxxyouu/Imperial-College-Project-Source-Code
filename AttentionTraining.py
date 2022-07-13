
"""
Script for Explanation-driven Model Training
"""

import os
import argparse
import torch
import os
from skresnet import skresnext50_32x4d
from layers import *
from RelevanceCamUtils import *
import Constants
import argparse
from torchvision import transforms
from Helper import *
from EvaluatorUtils import *
import torch.optim as optim
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# set the seed for reproducibility
rng_seed = 99
torch.manual_seed(rng_seed)
print('Device being used: {}'.format(Constants.DEVICE))


default_model_name = 'skresnext50_32x4d'
my_parser = argparse.ArgumentParser(description='')
my_parser.add_argument('--model_name',
                        type=str, default=default_model_name,
                        help='main_model name to be used for main_model retrival and weight replacement') 
my_parser.add_argument('--batch_size',
                        type=int, default=2,
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
                        type=bool, action=argparse.BooleanOptionalAction, # example: ckpt_epoch_500
                        help='pretrain argument')      
my_parser.add_argument('--cam',
                        type=str, default='relevance-cam',
                        help='select a cam') 
my_parser.add_argument('--alpha',
                        type=float, default=1,
                        help='alpha for relevance cam')
my_parser.add_argument('--pickel_name',
                        type=str, default='{}_AttentionTrained'.format(default_model_name),
                        help='pickel name') 
my_parser.add_argument('--target_layers',
                        type=str, default='3,4',
                        help='use comma to split the layer number')                      
args = my_parser.parse_args()

# Sanity checks for the script arguments
print('Model Name: {}'.format(args.model_name))
args.pickel_name = '{}_targetLayers{}_alpha{}_{}'.format(args.pickel_name, args.target_layers, args.alpha, args.cam)
print('Pickel Name: {}'.format(args.pickel_name))
print('Batch Size: {}'.format(args.batch_size))
print('Epochs Training {}'.format(args.epochs))
print('Early Stopping Patience {}'.format(args.earlyStoppingPatience))
print('Learning Rate {}'.format(args.learningRate))
print('CAM: {}'.format(args.cam))
if not args.pretrain:
    args.pretrain = True
print('Pretrain {}'.format(args.pretrain))
print('alpha {}'.format(args.alpha))
args.target_layers = [int(layerNum) for layerNum in args.target_layers.split(',')] # conver to numpy array for indexing
print('Target Layers {}'.format(args.target_layers))

# NOTE: MODIFY AND LOAD THE MODEL

def create_model(args):
    model = skresnext50_32x4d(pretrained=args.pretrain)
    model.num_classes = 2 #NOTE required to do CLRP and SGLRP
    model.fc = Linear(model.fc.in_features, model.num_classes, device=Constants.DEVICE, dtype=Constants.DTYPE)
    model.to(Constants.DEVICE)
    return model

main_model = create_model(args)
aux_model = create_model(args) # NOTE: Produce explanation, evaluation only
print('Model successfully loaded')

# NOTE: DATA TRANSFORMATION
train_transforms = transforms.Compose(
    [
        transforms.ToTensor(), 
        transforms.CenterCrop(230), # transforms.CenterCrop((336, 350)), 230 is the number that has the largest square in a circle
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation((0, 270)),
        transforms.RandomAutocontrast(0.25),
        transforms.Normalize(
        [Constants.DATA_MEAN, Constants.DATA_MEAN, Constants.DATA_MEAN], 
        [Constants.DATA_STD,Constants.DATA_STD, Constants.DATA_STD])
    ]
)
test_transforms = transforms.Compose([
    transforms.ToTensor(), 
    transforms.CenterCrop(230),
    transforms.Normalize(
        [Constants.DATA_MEAN, Constants.DATA_MEAN, Constants.DATA_MEAN], 
        [Constants.DATA_STD,Constants.DATA_STD, Constants.DATA_STD]
    )
])

#NOTE: NECESSARY TRAINING VARIABLES
data_dict = pytorch_dataset(args.batch_size, train_transforms, test_transforms)
train, train_dataloader = data_dict['train']
val, val_dataloader = data_dict['val']
test, test_dataloader = data_dict['test']

optimizer = optim.Adamax(main_model.parameters(), lr=args.learningRate, weight_decay=1e-8)
    
loss_func = nn.CrossEntropyLoss()

model_weights_des = Constants.SAVED_MODEL_PATH

value = dict()
def forward_hook(module, input, output):
    value['activations'] = output
def backward_hook(module, input, output):
    value['gradients'] = output[0]

def register_hooks(model):
    layer1_forward_handle = model.layer1.register_forward_hook(forward_hook)
    layer2_forward_handle = model.layer2.register_forward_hook(forward_hook)
    layer3_forward_handle = model.layer3.register_forward_hook(forward_hook)
    layer4_forward_handle = model.layer4.register_forward_hook(forward_hook)
    
    layer1_backward_handle = model.layer1.register_full_backward_hook(backward_hook)
    layer2_backward_handle = model.layer2.register_full_backward_hook(backward_hook)
    layer3_backward_handle = model.layer3.register_full_backward_hook(backward_hook)
    layer4_backward_handle = model.layer4.register_full_backward_hook(backward_hook)
    f_hooks = [layer1_forward_handle, layer2_forward_handle, layer3_forward_handle, layer4_forward_handle]
    b_hooks = [layer1_backward_handle, layer2_backward_handle, layer3_backward_handle, layer4_backward_handle]  
    return f_hooks, b_hooks

if args.cam == 'relevance-cam':
    aux_f_hooks, aux_b_hooks = register_hooks(aux_model)
    print('Registered Hooks')

def check_accuracy(model, loader, best_model=False, store_sample=False, _print=False):
        # function for test accuracy on validation and test set
        
        # if best_model:
        #     trained_model_loc = os.path.join(Constants.SAVED_MODEL_PATH, '{}.pt'.format(self.model_name))
        #     self.model_wrapper.load_learned_weights(trained_model_loc)
        #     print('Finished Loading the Best Model for {}'.format(args.model_name))

        num_correct = 0
        num_samples = 0
        acc, losses = 0, []

        model.eval()  # set model to evaluation mode
        with torch.no_grad():
            for x, y, names in loader:

                x = x.to(device=Constants.DEVICE, dtype=Constants.DTYPE)  # move to device
                y = y.to(device=Constants.DEVICE, dtype=torch.long)

                _, scores = model(x)
                _, preds = scores.max(1)

                # record the loss
                losses.append(loss_func(scores, y))

                # if store_sample:
                #     print('Saving correctly classified images')
                #     self._extract_correct_preds_and_save(preds, y, x, names)

                num_correct += (preds == y).sum()
                num_samples += preds.size(0)

        acc = float(num_correct) / num_samples

        if _print:
            print('Accuracy: {}'.format(acc))
        return float(acc), sum(losses)/len(losses)

#NOTE: MAIN LOOP TRAINING
print('Training Started')
patience, optimal_val_loss = args.earlyStoppingPatience, np.inf
for e in range(args.epochs):
    for t, (x, y, name) in enumerate(train_dataloader):
        optimizer.zero_grad()
        x = x.to(device=Constants.DEVICE, dtype=Constants.DTYPE)  # move to device, e.g. GPU
        y = y.to(device=Constants.DEVICE, dtype=torch.long)

        # run the auxilary network first to produce explanations
        targets = y.tolist()
        if args.cam == 'relevance-cam':
            all_cams, logits = aux_model(x, target_class=targets, mode='all')
            all_cams = [max_min_lrp_normalize(cam) for cam in all_cams]
            # filter the layer cams that match the user input
            cams = {}
            for i, cam in enumerate(all_cams):
                layerNum = i + 1
                if layerNum in args.target_layers:
                    cams[i] = cam

        else:
            cams = {}
            layers = [aux_model.layer1, aux_model.layer2, aux_model.layer3, aux_model.layer4]
            targets = [ClassifierOutputTarget(target) for target in targets]
            for i, layer in enumerate(layers):
                # generate the cam for layers that are specified by the script argument
                layerNum = i + 1
                if layerNum in args.target_layers:
                    cam = switch_cam(args.cam, aux_model, [layer])
                    results, logits = cam(input_tensor=x, targets=targets, scale=False, retain_model_output=True)
                    cams[i] = results
        
        # NOTE: replace the CAM with zero if it is wrongly classified
        correct_class = torch.argmax(logits, dim=1)
        for layer in cams:
            cams[layer][(correct_class != y), :] = 0
            # print('Amount of attention being used: {}'.format(torch.sum(correct_class == y).item()))
            # sanity check to see how many of the mare non-zero out
            if torch.all(correct_class != y):
                assert(torch.sum(cams[layer]) == 0)
            if torch.any(correct_class == y):
                assert(torch.sum(cams[layer]) != 0)
            #make sure the cam values are between zero and one (normalized)
            assert(torch.max(cams[layer]) <= 1)
            assert(torch.min(cams[layer]) >= 0)        
    
        # run the main network and attend using the cam results from the auxilary network
        main_model.train()  # put main_model to training mode
        _, unnormalized_score = main_model(x, attendCAM=cams, alpha=args.alpha) # unnormalized
        loss = loss_func(unnormalized_score, y)

        # Update the parameters of the main_model using the gradients
        loss.backward()
        optimizer.step()

        # replace the weights of the auxilary network after each iteration
        aux_model.load_state_dict(main_model.state_dict())

        if t % 20 == 0:
            print('{}/{} Iteration loss: {}'.format(t, len(train_dataloader), loss))

    # evaluate the validation dataset after every epoch
    val_acc, val_loss = check_accuracy(main_model, val_dataloader)
    print('Epoch: {}, Validation Loss {}, val_acc {}'.format(e, val_loss, val_acc))

    if val_loss < optimal_val_loss:
        print('Saving main_model')
        # save the main_model to destination
        model_dest = os.path.join(model_weights_des, '{}.pt'.format(args.model_name))
        torch.save(main_model.state_dict(), model_dest)

        # udpate the tracking parameters
        optimal_val_loss = val_loss
        patience = args.earlyStoppingPatience
    else:
        patience -= 1

    # stop training when epoch acc no longer improve for consecutive {patience} epochs
    if patience <= 0:
        break

if args.cam == 'relevance-cam':
    for hook in aux_f_hooks:
        hook.remove()
    for hook in aux_b_hooks:
        hook.remove()





