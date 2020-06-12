import numpy as np 
import torch
from torchvision import transforms, datasets,models
from PIL import Image 
import matplotlib.pyplot as plt
import os
import json


def process_image(image):
    
    
    
    img = Image.open(image)
    
    transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()])
                                  
    
                                  
    transformed_img = transform(img)
                                   
    img = np.array(transformed_img)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image = (np.transpose(img,(1,2,0)) - mean) / std 
    image = np.transpose(image, (2, 0, 1))
    
    return image
                                   
                            
                                   
def data_loader(data_dir):
                                   
    train_dir = data_dir +'/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
                                   
    data_transforms = transforms.Compose([transforms.Resize(255), 
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485,0.456,0.406],
                                                                  [0.229, 0.224, 0.225])])
                                          
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                            transforms.Resize(255), 
                                            transforms.CenterCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485,0.456,0.406],
                                                                  [0.229, 0.224, 0.225])])
    
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform= data_transforms)
    test_data = datasets.ImageFolder(test_dir, transform= data_transforms)
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size = 64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 64)
    
    return train_data, valid_data, test_data, trainloader, validloader, testloader


def resultdisplay(image, probs, classes, top_k):
    #show image
    fig, ax = plt.subplots()
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = np.transpose(image,(1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    
    #show probabilities bargraph
    fig, ax = plt.subplots()
    
    labels =[]
    for cl in classes:
        labels.append(cat_to_name[cl])
        
        
    ax.barh(labels,width=probs)
    ax.set_aspect(0.1)
    ax.set_yticks(np.arange(top_k))
    
    ax.set_title('Class Probability')
    
    plt.tight_layout()
    plt.show()
       
    return ax

def get_class(classes,  category_names):
    
    names = []
    
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    for i in classes:
        names.append(cat_to_name[i])
        
    return names
    

    

    
    
def show_classes(prob, classes, top_k):
    
    print('--------Predictions for Image--------')
    i = 0
    while (i < top_k):
        print('%*s. Class: %*s. Pr= %.4f'%(7, i+1, 3, classes[i], prob[i]))
        i += 1
                                   
            
def network_model (arch):
    model = getattr(models, arch)(pretrained=True)
    return model
    
  
    
    