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
    args = get_input_args()
    print(args)
    data_directory = get_directory(args)
    print('directory is',data_directory)
    transformed_data = transform_data(data_directory)
    print('transfomed data is ',transformed_data)
    model = databuilder(args)
    print('model is',model)
    model = train(model,transformed_data,args)
    print('trained model is ',model)
    save_check_point(model,transformed_data,args)
    print("done")
    

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir',default="./flowers/",help='Datasets directory')
    parser.add_argument('--save_dir', default='checkpoint.pth', help='Checkpoint save directory')
    parser.add_argument('--arch', choices=[ 'vgg16','vgg13'], default='vgg16', help='neural network')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate value')
    parser.add_argument('--hidden_units', type=int, nargs='+', default=[4096,1024], help='Hidden units')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--gpu', default = 'cuda' ,action="store_true", help='GPU')
    args = parser.parse_args()

    return args

def get_directory(args):
    args.data_dir = "./flowers"
    train_dir = args.data_dir + '/train'
    valid_dir = args.data_dir + '/valid'
    test_dir = args.data_dir + '/test'
    data_directory = [train_dir,test_dir,valid_dir]
    return data_directory


def transform_data(data_dir):
    train_dir, test_dir, valid_dir =  data_dir
    data_transforms =transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], 
                                                          [0.229, 0.224, 0.225])])

    #validating 
    valid_transofrms = transforms.Compose([transforms.Resize(224),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], 
                                                          [0.229, 0.224, 0.225])])
    #training
    train_transforms =transforms.Compose([transforms.Resize(255),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    #testing
    test_transforms =transforms.Compose([transforms.Resize(224),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], 
                                                                  [0.229, 0.224, 0.225])])
    # Load the datasets with ImageFolder
    valid_datasets = datasets.ImageFolder(valid_dir, transform=data_transforms)
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)


    # Using the image datasets and the trainforms, define the dataloaders
    validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=32,shuffle=True)
    trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size=64,shuffle=True)
    testloaders = torch.utils.data.DataLoader(test_datasets, batch_size=64)
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    loaders_dict = {
               'validloaders':validloaders,
               'trainloaders':trainloaders,
               'testloaders':testloaders,
               'cat_to_name':cat_to_name,
               'train_datasets':train_datasets}
    return loaders_dict


def databuilder(args):
    if (args.arch =='vgg' or args.arch =='vgg13' or args.arch =='vgg16'):
        model = models.vgg16(pretrained=True) #test it later for models.args.arch 
        inputs=25088
    #freezing
    for param in model.parameters():
        param.requires_grad = False
        
    hidden_units = args.hidden_units
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(inputs, 4096)),
                          ('relu1', nn.ReLU()),
                          ('do1', nn.Dropout(0.5)),
                          ('fc2', nn.Linear(4096, 1024)),
                          ('relu2', nn.ReLU()), 
                          ('do2', nn.Dropout(0.5)),
                          ('fc3', nn.Linear(1024,102)), 
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier = classifier    
    return model
    
def validation(model, optimizer,testloader, criterion,args):
    test_loss = 0
    accuracy = 0
    device=args.gpu
    for images, labels in testloader:
        optimizer.zero_grad()
        images, labels = images.to(device) , labels.to(device)
        model.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy
  
def train(model,data,args):
    lr = float(args.learning_rate)
    epochs = args.epochs
    testloader =data['testloaders']
    trainloader=data['trainloaders']
    validloader=data['validloaders']
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
  
    steps = 0
    print_every = 40
    device = args.gpu
    for e in range(epochs):
        running_loss = 0
        model.to(device)
        for ii,(inputs, labels) in enumerate(trainloader):
            steps += 1
            inputs,labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model.forward(inputs)

            loss = criterion(output, labels)

            loss.backward()

            optimizer.step()  

            running_loss += loss.item()      

            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    test_loss, accuracy = validation(model,optimizer,testloader, criterion,args)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
                running_loss = 0
                model.train()
    return model

                           
def save_check_point(model,data,args):
    save_dir = args.save_dir
    train_datasets= data['train_datasets']
    model.class_to_idx = train_datasets.class_to_idx
    model.to('cpu')
    checkpoint = {'input_size': 25088,
              'output_size': 102,
              'hidden_layer1':4096,
              'hidden_layer2':1024,
              'index_to_class': {i: j for i, j in train_datasets.class_to_idx.items()},
              'state_dict': model.state_dict()}
    torch.save(checkpoint, save_dir)
    return
                           
                            
        

if __name__ == "__main__":
    main()    
        
        


