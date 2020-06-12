import argparse
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import Model
import utils
import time
import pdb
 



#pdb.set_trace()


parser = argparse.ArgumentParser()

parser.add_argument('data_dir', action='store', help='directory containing images')
parser.add_argument('--save_dir', action='store', help='save trained checkpoint to this directory')
parser.add_argument('--arch', action='store', help='what kind of pretrained architecture to use', default='vgg13')
parser.add_argument('--input_units', action='store', help='# of hidden units to add to model', type=int, default=20)
parser.add_argument('--gpu', action='store_true', help='use gpu to train model')
parser.add_argument('--epochs', action='store', help='# of epochs to train', type=int, default=20)
parser.add_argument('--hidden_units', action='store', help='# of hidden units to add to model', type=int, default=1024)
parser.add_argument('--output_units', action='store', help='# of classes to output', type=int, default=102)

args=parser.parse_args()

#Sorting the data for training, validation, testing

data_dir = 'flowers'

train_data, valid_data, test_data, trainloader, validloader, testloader = utils.data_loader(data_dir)


#Create Model 

model = utils.network_model(args.arch)

#model = models.vgg13(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

    
#input_units = utils.get_input_units(model, args.arch)

model.classifier = Model.network( args.output_units,args.hidden_units,drop_p=0.5) 

#print(model)
 
 # train  Model 

optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
criterion = nn.NLLLoss()



Model.train_network(model, trainloader, validloader, optimizer,criterion, args.epochs, args.gpu)

# testing Model 

test_loss, test_accuracy = Model.test_network(model, testloader, criterion, args.gpu)

print("\n ---\n Test Loss: {:.4f}".format(test_loss), "Test Accuracy: {:.4f}".format(test_accuracy) )

# save netowrk 
# npdb.set_trace()

Model.save_checkpoint(model, train_data, optimizer, args.save_dir, args.arch)


