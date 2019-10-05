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
    parser.add_argument('--gpu',default = 'cuda',action="store_true", help='GPU')
    args = parser.parse_args()
    return args


def load_check_point(args):
    checkpoint = torch.load(args.checkpoint)
    classifier =nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(checkpoint['input_size'], checkpoint['hidden_layer1'])),
                          ('relu1', nn.ReLU()),
                          ('do1', nn.Dropout(0.5)),
                          ('fc2', nn.Linear(checkpoint['hidden_layer1'],checkpoint['hidden_layer2'])),                                                            ('relu2', nn.ReLU()), 
                          ('do2', nn.Dropout(0.5)),
                          ('fc3', nn.Linear(checkpoint['hidden_layer2'],checkpoint['output_size'])), 
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier = classifier
    model.class_to_idx =  checkpoint['index_to_class']
    model.load_state_dict(checkpoint['state_dict'])
    return model
def process_image(image):
    
    pil_image = Image.open(image)
    transformed_image =transforms.Compose([transforms.Resize(224),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    return transformed_image(pil_image)
                        


def predict(image_path, model,args):
    
    topk = args.top_k
    device = args.gpu
    model.to(device)
    image = process_image(image_path)
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
    device = args.gpu
    category_names = args.category_names
    probs = predict(path, model,args)
    ps = np.array(probs[0][0])
    print('probs are ',np.array(probs[0][0]))
    print('labels are ',np.array(probs[1][0]))
    locs = [cat_to_name[str(index + 1)] for index in np.array(probs[1][0])]
    
    for index in range(topk):
        print('the picture has a probability of {}  to be a  {} '.format(ps[index],locs[index]))
        
    return 
    
    
    
    
model = models.vgg16(pretrained=True)    

main()        
