import argparse
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json 
import Model
import utils 
import os
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('input', action='store', help='path to image to be classified')
parser.add_argument('checkpoint', action='store', help='path to stored model')
parser.add_argument('--top_k', action='store', type=int, default=5, help='how many most probable classes to print out')
parser.add_argument('--category_names', action='store',default ="cat_to_name.json", help='file which maps classes to names')
parser.add_argument('--gpu', action='store_true', help='use gpu to infer classes')
args=parser.parse_args()



if args.gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else: device = "cpu"
    
model = Model.load_model(args.checkpoint, args.gpu)


img = utils.process_image(args.input)#.to(device) 

probs, classes = Model.predict(img, model, args.top_k)



if(args.category_names != None): 
    classes = utils.get_class(classes, args.checkpoint, args.category_names)
else:
    classes=utils.get_class(classes, args.checkpoint, None)
    

#utils.resultdisplay(img, probs, classes, args.top_k)

utils.show_classes(probs, classes, args.top_k)




    



