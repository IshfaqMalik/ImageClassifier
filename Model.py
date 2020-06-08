import torch



from torch import nn, optim

from torchvision import models
from collections import OrderedDict
import torch.nn.functional as F
import time
import utils
import numpy as np


    
def network (input_units, output_units, hidden_units, drop_p):
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_units, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(p = drop_p)),
    
        ('fc2', nn.Linear(hidden_units, output_units)),
        ('output', nn.LogSoftmax(dim=1))]))
    
    return classifier
    
    
def train_network(model, trainloader, validloader,optimizer,criterion, epochs, gpu):
    
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
    
    
    
    model.to(device)
    model.train()
    
  
    
    
  #  validation_step = True
    
    print_every = 30
    steps = 0
    
    
    for epoch in range(epochs):
        print("Epoch")
        
        training_loss = 0
        
        for ii, (inputs,labels) in enumerate(trainloader):
            print("Step")
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            training_loss = training_loss + loss.item()
            
            #checking validation 
            
            if steps % print_every == 0:
           
                valid_loss = 0
                valid_accuracy = 0
                model.eval()             
               
                
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps,labels)
                        valid_loss = valid_loss + batch_loss.item()
                        
                        #checking accuracy
                        probs = torch.exp(logps)
                        top_p, top_class = probs.topk(1, dim = 1)
                        equals = top_class == labels.view(*top_class.shape)
                        valid_accuracy = valid_accuracy + torch.mean(equals.type(torch.FloatTensor)).item()
                        
                                             
    
                print('\nEpoch: {}/{} '.format(epoch + 1, epochs),
                      '\n    Training:\n      Loss: {:.4f}  '.format(training_loss / print_every))
    
                print("\n    Validation:\n      Loss: {:.4f}  ".format(valid_loss / len(validloader)),
                      "Accuracy: {:.4f}".format(valid_accuracy / len(validloader)))
    
                training_loss = 0
                model.train()
   


def save_checkpoint(model,train_data,optimizer, save_dir,arch):
    
    
    
    model_checkpoint = {'arch':arch,
                        
                        
                        'learning_rate': 0.003,       
                        'batch_size': 64,
                        'classifier' : model.classifier,
                        'optimizer': optimizer.state_dict(),
                        'state_dict': model.state_dict(),
                        'class_to_idx': train_data.class_to_idx,}
    if(save_dir == None): torch.save(model_checkpoint, 'checkpoint.pth')
    else:torch.save(model_checkpoint, save_dir+'checkpoint.pth')
        
        
def load_model (file_path, gpu):
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else: 
        device = "cpu"
        
    checkpoint = torch.load(file_path)
    learning_rate = checkpoint['learning_rate']
    model = utils.network_model(checkpoint['arch'])
    model.classifier = checkpoint['classifier']
    model.optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model


def predict(image_path, model, topk):
  
    cuda = torch.cuda.is_available()
    if cuda:
        # Move model parameters to the GPU
        model.cuda()
        print("Number of GPUs:", torch.cuda.device_count())
        print("Device name:", torch.cuda.get_device_name(torch.cuda.device_count()-1))
    else:
        model.cpu()
        print("We go for CPU")
    
    # turn off dropout
    model.eval()

    
    
    
    # tranfer to tensor
    image = torch.from_numpy(np.array([image_path])).float()
    
    # The image becomes the input
    
    if cuda:
        image = image.cuda()
        
    output = model.forward(image)
    
    probabilities = torch.exp(output).data
    
    # getting the topk (=5) probabilites and indexes
    # 0 -> probabilities
    # 1 -> index
    prob = torch.topk(probabilities, topk)[0].tolist()[0] # probabilities
    index = torch.topk(probabilities, topk)[1].tolist()[0] # index
    
    ind = []
    for i in range(len(model.class_to_idx.items())):
        ind.append(list(model.class_to_idx.items())[i][0])

    # transfer index to label
    label = []
    for i in range(5):
        label.append(ind[index[i]])

    return prob, label
    
    
 #Testing 

def test_network (model,loader, criterion,gpu):
        if gpu:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
         device = "cpu"
        
        test_loss = 0
        test_accuracy = 0
        
        model.eval()
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            loss = criterion(logps,labels)
            test_loss = test_loss + loss.item()
    
            probabilities = torch.exp(logps)
            top_p, top_class = probabilities.topk(1, dim =1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy = test_accuracy + torch.mean(equals.type(torch.FloatTensor)).item()
            
            return test_loss/len(loader), test_accuracy/len(loader)





                        
                        
                        
                
            
   
    
        
    
        
        
    