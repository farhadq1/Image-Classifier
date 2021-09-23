#Load packages
import argparse
import pandas as pd
import numpy as np
from collections import OrderedDict
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import json

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', action='store', help='Directory for images')
parser.add_argument('--save_dir', action='store', help='Checkpoint save location',default = '/home/workspace/ImageClassifier/' )
parser.add_argument('--arch', action='store', help='Choose from vgg19 or vgg13', default='vgg19')
parser.add_argument('--gpu', action='store_true', help='Specify gpu to be used',default="gpu")
parser.add_argument('--epochs', action='store', help='Number of epochs', type=int, default=8)
parser.add_argument('--learning_rate', action='store', help='Specify learning rate', type=float, default=0.01)
parser.add_argument('--hidden_units', action='store', help='Specify hidden units', type=int, default=1024)
parser.add_argument('--output_size', action='store', help='Specify output units', type=int, default=102)

args=parser.parse_args()
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
train_transforms = transforms.Compose([transforms.RandomRotation(35),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])
                                     ])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])
                                     ])

test_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], 
                                                          [0.229, 0.224, 0.225])
                                     ])


#Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
#Using the image datasets and the transforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
model_arch = args.arch
#create model using --arch argument with vgg19 as default
if args.gpu:
    set_dev = "cuda"
device = torch.device(set_dev if torch.cuda.is_available() else "cpu")
#print(device)
if model_arch == 'vgg19':
    model = models.vgg19(pretrained=True)
elif (model_arch == 'vgg13'):
    model = models.vgg13(pretrained=True)

for param in model.parameters():
    param.requires_grad = False
#params are now frozen so that we do not backprop thru them again


classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, args.hidden_units)),
                          ('drop', nn.Dropout(p=0.45)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(args.hidden_units, args.output_size)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier

#Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.classifier.parameters(), lr=args.learning_rate)

model.to(device)
#train model
epochs = args.epochs
steps = 0
running_loss = 0
print_every = 15
print("Training the network")
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    valid_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()
print("Testing the network")
#test model
test_loss = 0
accuracy = 0
with torch.no_grad():
    model.eval()
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)

        test_loss += batch_loss.item()

        # Calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Test accuracy: {accuracy/len(testloader):.3f}")

#save network to checkpoint
model.class_to_idx  = train_data.class_to_idx
checkpoint = {'input_size': 25088,
              'output_size': args.output_size,
              'epochs': epochs,
              'learning_rate': args.learning_rate,
              'batch_size': 64,
              'classifier' : classifier,
              'optimizer': optimizer.state_dict(),
              'arch': args.arch,
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx}

torch.save(checkpoint,args.save_dir+'/'+'checkpoint.pth')