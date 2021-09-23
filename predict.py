#Import packages
from torch import optim
import torch
from torch import nn
import pandas as pd
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
import torchvision
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import argparse
import json
from PIL import Image
import os, random

parser = argparse.ArgumentParser()
parser.add_argument('input', action='store', help='image folder number and filename to be classified e.g. /1/image_06743.jpg')
parser.add_argument('checkpoint', action='store', help='path to stored model')
parser.add_argument('--top_k', action='store', type=int, default=1, help='Probability rank')
parser.add_argument('--category_names', action='store', help='File mapping image to classes',default = 'cat_to_name.json')
parser.add_argument('--gpu', action='store_true', help='use gpu')
args=parser.parse_args()

#load classifier
def load_checkpoint(path):
    checkpoint = torch.load(path)
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = optim.SGD(model.classifier.parameters(), lr=0)
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer
#Process image data
def process_image(image_input):
    
    img = Image.open(image_input)
    img = img.resize((256,256))
    value_set = 12 #set value for adjusting 
    img = img.crop((value_set,value_set,256-value_set,256-value_set))
    img= np.array(img)/255
    img = (img - np.array([0.485, 0.456, 0.406]))/np.array([0.229, 0.224, 0.225])
    return img.transpose(2,0,1)
#Set up prediction
def predict(image_path, model, topk,processor):
    if processor:
        set_dev = "cuda"
    device = torch.device(set_dev if torch.cuda.is_available() else "cpu")
    model.to(device)
    #Set to evaluation 
    model.eval()
    #process the image
    image = process_image(image_path)
    #Setting up the image as input
    image = Variable(torch.from_numpy(np.array([image])).float())    
    image = image.to(device)        
    output = model.forward(image)
    #Extract probabilities and labels
    probabilities = torch.exp(output).data
    probs = torch.topk(probabilities, topk)[0].tolist()[0]
    index = torch.topk(probabilities, topk)[1].tolist()[0]
    indx = []
    for ind in range(len(model.class_to_idx.items())):
        indx.append(list(model.class_to_idx.items())[ind][0])

    labels = []
    for i in range(topk):
        labels.append(indx[index[i]])

    return probs, labels

with open(args.category_names, 'r') as f:
    catname = json.load(f)
trained_model,optimizer = load_checkpoint(args.checkpoint) 
img_path = './flowers/test' + args.input
prob, classes = predict(img_path, trained_model,args.top_k,args.gpu)    

print(prob)
print(classes)
print([catname[a] for a in classes])
