import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
import torch
import argparse
import os
import json
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms,models
from PIL import Image

def main():
    args = parse()
    model = load_check_point(args)
    print('model is' , model)
    show_prediction(args.input_path,model,args)
    print('done')
    
    return 


def parse():
    print('start')
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path',type = str ,default = './flowers/test/23/image_03382.jpg', help='path to image input')
    parser.add_argument('checkpoint',type = str, default='./checkpoint.pth',help = 'checkpoint path')
    parser.add_argument('--top_k',type=int, default=5, action="store",help= 'number of topk')
    parser.add_argument('--category_names',type=str ,default='./cat_to_name.json',help = 'category name')
    parser.add_argument('--gpu',default = 'false',action="store_true", help='GPU')
    args = parser.parse_args()
    return args


def load_check_point(args):
    checkpoint = torch.load(args.checkpoint)
    classifier =nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(checkpoint['input_size'], checkpoint['hidden_layer'])),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(checkpoint['hidden_layer'],checkpoint['output_size'])), 
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    architecture = checkpoint['architecture']
    if architecture == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
    
    model.classifier = classifier
    model.class_to_idx =  checkpoint['index_to_class']
    model.load_state_dict(checkpoint['state_dict'])
    return model
def process_image(image):
    
    pil_image = Image.open(image)
    width, height = pil_image.size
    
    if height > width:
        height = int(max(height * 256 / width, 1))
        width = 256
    else:
        width = int(max(width * 256 / height, 1))
        height = 256
        
        
    x0 = (width - 224) / 2
    y0 = (height - 224) / 2
    x1 = (width + 224) / 2
    y1 = (height + 224) / 2  
    
    new_size = pil_image.resize((width, height))
    pil_image = new_size.crop((x0, y0, x1, y1))  
    np_image = np.array(pil_image)
    np_image = (np_image - np_image.mean()) / np_image.std()
    np_image = np_image.transpose(2, 0, 1)
    
    
    return np_image
                        


def predict(image_path, model,args):
    
    topk = args.top_k
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    model.to(device)
    image = process_image(image_path)
    image = torch.from_numpy(image)
    image.unsqueeze_(0)
    image = image.float().to(device)
    with torch.no_grad():
        output = model.forward(image)
    ps = F.softmax(output,dim=1)
    return ps.topk(topk)


def show_prediction(path,model,args):
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    topk= args.top_k
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    category_names = args.category_names
    
    probs= predict(path, model,args)
    ps = np.array(probs[0][0])
    idx_to_class = {val: key for key, val in model.class_to_idx.items()} 
    locs = [cat_to_name[idx_to_class[idx]] for idx in np.array(probs[1][0])]
    
    for index in range(topk):
        print('the picture has a probability of {}  to be a  {} '.format(ps[index],locs[index]))
        
    return 
    
    
    
    

main()        
