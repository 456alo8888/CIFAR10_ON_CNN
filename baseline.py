
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory

from torch.utils.data import Dataset
from torchvision import datasets 
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt

import argparse
import os
from pathlib import Path
import numpy as np
import time
from torch.utils.data import DataLoader

epoch_num = 3
number_layers = 3
batch_size_train = 16
batch_size_test = 16
l_r = 0.01
momentum = 0.9
log_interval = 100  #how often the training result being printed (1 time for every 100 batches)
random_seed = 318
torch.manual_seed(random_seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# I got error on downloading this dataset so I only use it online without downloading


# def free_gpu_memory():
#     torch.cuda.empty_cache()
#     gc.collect()

def free_then_unfreeze_layers_efficientnet(model, num_unfreeze_layers):
    # Freeze all model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last 'num_unfreeze_layers' layers
    layers = [
        model._blocks[-1],
        model._blocks[-2],
        model._conv_head,
        model._bn1,
        model._fc
    ]
    unfrozen_layers_count = 0

    for layer in reversed(layers):
        if unfrozen_layers_count >= num_unfreeze_layers:
            break
        for param in layer.parameters():
            param.requires_grad = True
        unfrozen_layers_count += 1

    return model


def preprocessing_data(transform_train , transform_test):
    #transform data into Tensor and normalize with mean = 0.5, standard deviation = 0.5 -> do lech chuan
    training_data = torchvision.datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform= transform_train
    )

    training_data , valid_data , remainder_data = torch.utils.data.random_split(training_data , [0.5, 0.1 , 0.4])

    valid_data.dataset.transform = transform_test

    test_data = torchvision.datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform= transform_test
    ) 

    return training_data , valid_data , test_data


def unfreeze_layers(model, num_unfreeze_layers=number_layers):

    # Freeze all model parameters
    for param in model.parameters():
        param.requires_grad = False

    """
    Gradually unfreeze layers starting from the deepest layers.
    `num_unfreeze_layers` specifies how many layers to unfreeze.
    """
    layers = list(model.children())

    # Helper function to recursively unfreeze parameters in a module
    def unfreeze_module(module):
        for param in module.parameters():
            param.requires_grad = True

    # Reverse the order to start unfreezing from the last layers
    layers.reverse()
    unfrozen_layers_count = 0

    # Unfreeze the last 'num_unfreeze_layers' layers
    for layer in layers:
        if unfrozen_layers_count >= num_unfreeze_layers:
            break
        unfreeze_module(layer)
        unfrozen_layers_count += 1

    return model



def unfreeze_layers_vgg16(model, num_unfreeze_layers=number_layers):

    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last 'num_unfreeze_layers' layers
    layers = list(model.features.children()) + list(model.classifier.children())
    layers.reverse()
    unfrozen_layers_count = 0

    for layer in layers:
        if unfrozen_layers_count >= num_unfreeze_layers:
            break
        for param in layer.parameters():
            param.requires_grad = True
        unfrozen_layers_count += 1

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Number of trainable parameters: {num_trainable_params}")

    return model




def change_last_layer(model, num_classes = 10):
    if isinstance(model, models.ResNet):
        # For ResNet models
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif isinstance(model, models.VGG):
        # For VGG models
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    elif isinstance(model, models.DenseNet):
        # For DenseNet models
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    elif isinstance(model, models.MobileNetV2):
        # For MobileNetV2 models
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif isinstance(model, models.EfficientNet):
        # For EfficientNet models
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif isinstance(model, models.Inception3):
        # For Inception models
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError("Model architecture not supported for automatic layer replacement")
    
    return model

    
# for shuffling the dataset when we sample a minibatch in the training phase 

def load_data(training_data , valid_data , test_data):
    train_dataloader = DataLoader(training_data , batch_size = batch_size_train , num_workers= 8,  shuffle= True)
    valid_dataloader = DataLoader(valid_data , batch_size = batch_size_test, num_workers= 8, shuffle= False)

    test_dataloader = DataLoader(test_data , batch_size= batch_size_test, num_workers = 8,  shuffle = False )
    data_loader = {'train' : train_dataloader , 'test':test_dataloader , 'valid':valid_dataloader}
    return data_loader

#after transform image become tensor and can only displayed using plt.imshow with img.squeeze()

def read_opts():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--folder", type=str, default="m3_d1_n10")
    # parser.add_argument("--model_params_path", type=str, default="best_model_params.pth")
    parser.add_argument("--baseline_model", type=str, default="resnet50")
    # parser.add_argument("--output", type=str, default="output.csv")
    
    
    options = vars(parser.parse_args())
    return options

def train_model(model , criterion , optimizer , scheduler ,best_model_params_path, data_loader,  num_epochs = 25):    
    since = time.time()
    best_acc = 0
    torch.save(model.state_dict(), best_model_params_path)
    best_acc = 0.0
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(num_trainable_params)
        
    if torch.cuda.is_available():
        print("cuda is on")
    
    for epoch in range(num_epochs):
        print(f"begin training epoch {epoch+1}/{num_epochs}")
        for phase in ['train' , 'valid']:
            running_loss = 0.0
            correct = 0 
            total = 0
            if phase == 'train':
                model.train()
            else:
                model.eval()

            for inputs, labels in data_loader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _ , prediction = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                total += labels.size(0)
                running_loss += loss.item()
                correct += torch.sum(prediction == labels.data)

            if phase == 'train':
                scheduler.step()
            epoch_loss = (running_loss)/total
            epoch_acc = correct / total 

            if phase == 'valid' and epoch_acc > best_acc:
                torch.save(model.state_dict(), best_model_params_path)
                best_acc = epoch_acc
        # print(f'Epoch [{epoch + 1}/{num_epochs}], training loss: {running_loss / len(train_dataloader)}, t loss: {}')

            if phase == 'train':
                print(f'Epoch [{epoch + 1}/{num_epochs}], training loss: {epoch_loss} , epoch time: {time.time() - since}')
            else:
                print(f' valid correct percent : {epoch_acc:.2f}')
        
                

    time_last = time.time() - since
    print(f'Training complete in {time_last // 60:.0f}m {time_last % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
    return model , time_last


def testing(model , dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() #use evaluation mode in order to stop using drop out
    correct = 0
    total = 0
    print("begin testing")
    test_dataloader = dataloader['test']
    with torch.no_grad(): #disable gradient calculating to saving memory
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device) #send to device to use cpu or gpu 
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1) #Gets the predicted class by finding the index of the maximum value in the output tensor along dimension 1
            total += labels.size(0)
            for i in predicted:
                if predicted[i] == labels[i]:
                    correct+=1

    # print(f'Accuracy of the ResNet50 on the 10000 test images: {100 * correct / total}%')
    return 100 * correct / total



def run_resnet50():
    transform_train = transforms.Compose(
    [transforms.RandomCrop(32, padding=4),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    transform_test = transforms.Compose([
        transforms.ToTensor() ,
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    training_data , valid_data , test_data = preprocessing_data(transform_train , transform_test)
    data_loader = load_data(training_data , valid_data , test_data)
    model = models.resnet50(weights= 'DEFAULT')
    # num_ftrs = model.fc.in_features
    # # Here the size of each output sample is set to 2.
    # # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
    # model.fc = nn.Linear(num_ftrs, 10)

    model = change_last_layer(model)

    #free 3 layers for train and freeze other layers
    model = unfreeze_layers(model , num_unfreeze_layers= 3)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    model , train_time = train_model(model = model, criterion = criterion  , optimizer = optimizer,scheduler= exp_lr_scheduler ,best_model_params_path='./modelweight/cifar_resnet50_best.pth',data_loader= data_loader, num_epochs= 12)
    result = testing(model , dataloader= data_loader)
    return result , train_time

def run_resnet101():
    transform_train = transforms.Compose(
    [transforms.RandomCrop(32, padding=4),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    transform_test = transforms.Compose([
        transforms.ToTensor() ,
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    training_data , valid_data , test_data = preprocessing_data(transform_train , transform_test)
    data_loader = load_data(training_data , valid_data , test_data)
    model = models.resnet101(weights= 'DEFAULT')
    # num_ftrs = model.fc.in_features
    # # Here the size of each output sample is set to 2.
    # # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
    # model.fc = nn.Linear(num_ftrs, 10)

    model = change_last_layer(model)
    model = unfreeze_layers(model , num_unfreeze_layers= 3)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    model , train_time = train_model(model = model, criterion = criterion  , optimizer = optimizer,scheduler= exp_lr_scheduler ,best_model_params_path='./modelweight/cifar_resnet101_best.pth',data_loader= data_loader, num_epochs= 12)
    result = testing(model , dataloader= data_loader)
    return result , train_time


def run_vgg16():
    
    transform_train = transforms.Compose([
        transforms.Resize(224),  # Resize images to 224x224
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Normalization for validation/testing
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    training_data , valid_data , test_data = preprocessing_data(transform_train, transform_test)
    dataloader = load_data(training_data , valid_data , test_data)

    model = models.vgg16(weights = 'DEFAULT')
    # num_ftrs = model.fc.in_features
    # # Here the size of each output sample is set to 2.
    # # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
    # model.fc = nn.Linear(num_ftrs, 10)

    model = change_last_layer(model)
    model = unfreeze_layers_vgg16(model , num_unfreeze_layers= 3)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    model, train_time = train_model(model = model, criterion = criterion  , optimizer = optimizer,scheduler= exp_lr_scheduler ,best_model_params_path='./modelweight/cifar_vgg16_best.pth',data_loader= dataloader, num_epochs= 12)
    result = testing(model , dataloader= dataloader)
    return result , train_time

def run_inception_v3():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform_train= transforms.Compose([
        transforms.RandomResizedCrop(299),  # Crop to 299x299 for InceptionV3
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(299),  # Resize the image to 299x299
        transforms.CenterCrop(299),  # Crop to 299x299
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    training_data , valid_data , test_data = preprocessing_data(transform_train, transform_test)
    dataloader = load_data(training_data , valid_data , test_data)

    
    model = models.inception_v3(weights= 'DEFAULT')
    # num_ftrs = model.fc.in_features
    # # Here the size of each output sample is set to 2.
    # # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
    # model.fc = nn.Linear(num_ftrs, 10)

    model = change_last_layer(model)

    model = unfreeze_layers(model , num_unfreeze_layers= number_layers)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    model, train_time = train_model(model = model, criterion = criterion  , optimizer = optimizer,scheduler= exp_lr_scheduler ,best_model_params_path='./modelweight/cifar_inception_v3_best.pth',data_loader= dataloader, num_epochs= 12)
    result = testing(model , dataloader)
    
    return result , train_time


def run_efficientnet_b0():
    # Mean and standard deviation for normalization (ImageNet statistics)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Transformations for train data
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),  # Crop to 224x224 (for all models except InceptionV3)
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Transformations for test data
    test_transform = transforms.Compose([
        transforms.Resize(256),  # Resize the smaller edge to 256
        transforms.CenterCrop(224),  # Crop to 224x224 (for all models except InceptionV3)
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    training_data , valid_data , test_data = preprocessing_data(train_transform , test_transform)
    dataloader = load_data(training_data , valid_data , test_data)

    model = models.efficientnet_b0(weights= 'DEFAULT')
    # num_ftrs = model.fc.in_features
    # # Here the size of each output sample is set to 2.
    # # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
    # model.fc = nn.Linear(num_ftrs, 10)
    model = change_last_layer(model)
    model = free_then_unfreeze_layers_efficientnet(model , num_unfreeze_layers= number_layers)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    model , train_time = train_model(model = model, criterion = criterion  , optimizer = optimizer,scheduler= exp_lr_scheduler ,best_model_params_path='./modelweight/cifar_efficientnet_b0_best.pth', data_loader= dataloader, num_epochs= 12)
    result = testing(model , dataloader)
    
    return result , train_time


def run_mobilenet_v2():
    # Mean and standard deviation for normalization (ImageNet statistics)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Transformations for train data
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),  # Crop to 224x224 (for all models except InceptionV3)
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Transformations for test data
    test_transform = transforms.Compose([
        transforms.Resize(256),  # Resize the smaller edge to 256
        transforms.CenterCrop(224),  # Crop to 224x224 (for all models except InceptionV3)
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    training_data , valid_data , test_data = preprocessing_data(train_transform , test_transform)
    dataloader = load_data(training_data , valid_data , test_data)

    model = models.mobilenet_v2(weights= 'DEFAULT')
    # num_ftrs = model.fc.in_features
    # # Here the size of each output sample is set to 2.
    # # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
    # model.fc = nn.Linear(num_ftrs, 10)
    model = change_last_layer(model)
    model = unfreeze_layers(model , num_unfreeze_layers= number_layers)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    model , train_time = train_model(model = model, criterion = criterion  , optimizer = optimizer,scheduler= exp_lr_scheduler ,best_model_params_path='./modelweight/cifar_mobilenet_v2_best.pth',data_loader= dataloader, num_epochs= 20)
    result = testing(model , dataloader)
    
    return result ,  train_time


def initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def run_mobilenet_v2_from_scratch():
    # Mean and standard deviation for normalization (ImageNet statistics)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Transformations for train data
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),  # Crop to 224x224 (for all models except InceptionV3)
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Transformations for test data
    test_transform = transforms.Compose([
        transforms.Resize(256),  # Resize the smaller edge to 256
        transforms.CenterCrop(224),  # Crop to 224x224 (for all models except InceptionV3)
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    training_data , valid_data , test_data = preprocessing_data(train_transform , test_transform)
    dataloader = load_data(training_data , valid_data , test_data)

    model = models.mobilenet_v2(weights=None)

    # # Replace the final classification layer for CIFAR-10 (10 classes)
    # model.classifier[1] = nn.Linear(model.last_channel, 10)
    model = change_last_layer(model)
    # Move the model to GPU
    model = model.cuda()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    model , train_time = train_model(model = model, criterion = criterion  , optimizer = optimizer,scheduler= exp_lr_scheduler ,best_model_params_path='./modelweight/cifar_mobilenet_v2_from_scratch.pth',data_loader= dataloader, num_epochs= 12)
    result = testing(model , dataloader)
    
    return result , train_time



# def run_resnet50_fromscratch():
#     model = models.resnet50(weights = None)
#     model.apply(initialize_weights)
#     model.fc = nn.Linear(model.fc.in_features , 10)
#     model = model.to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#     exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
#     model = train_model(model = model, criterion = criterion  , optimizer = optimizer,scheduler= exp_lr_scheduler ,best_model_params_path='./modelweight/cifar_densenet121_best.pth', num_epochs= 20)
#     result = testing(model)
#     return result

# def run_resnet50_from_scratch():
    
#     transform_train = transforms.Compose(
#     [transforms.RandomCrop(32, padding=4),
#      transforms.RandomHorizontalFlip(),
#      transforms.ToTensor(),
#      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

#     transform_test = transforms.Compose([
#         transforms.ToTensor() ,
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])

#     training_data , valid_data , test_data = preprocessing_data(transform_train , transform_test)
#     dataloader = load_data(training_data , valid_data , test_data)

#     model = models.resnet50(weights= None)
#     model.apply(initialize_weights)
#     num_ftrs = model.fc.in_features
#     # Here the size of each output sample is set to 2.
#     # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
#     model.fc = nn.Linear(num_ftrs, 10)
#     model = unfreeze_layers(model , num_unfreeze_layers= 2)
#     model = model.to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#     exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
#     model = train_model(model = model, criterion = criterion  , optimizer = optimizer,scheduler= exp_lr_scheduler ,best_model_params_path='./modelweight/cifar_resnet50_from_scratch.pth',data_loader= dataloader, num_epochs= 20)
#     result = testing(model , dataloader= dataloader)
#     return result




def run_densenet121():
    # Mean and standard deviation for normalization (ImageNet statistics)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Transformations for train data
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),  # Crop to 224x224 (for all models except InceptionV3)
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Transformations for test data
    test_transform = transforms.Compose([
        transforms.Resize(256),  # Resize the smaller edge to 256
        transforms.CenterCrop(224),  # Crop to 224x224 (for all models except InceptionV3)
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    training_data , valid_data , test_data = preprocessing_data(train_transform , test_transform)
    dataloader = load_data(training_data , valid_data , test_data)


    model = models.densenet121(weights= 'DEFAULT')
    # num_ftrs = model.fc.in_features
    # # Here the size of each output sample is set to 2.
    # # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
    # model.fc = nn.Linear(num_ftrs, 10)

    model = change_last_layer(model)
    model = unfreeze_layers(model , num_unfreeze_layers=number_layers)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    model , train_time = train_model(model = model, criterion = criterion  , optimizer = optimizer,scheduler= exp_lr_scheduler ,best_model_params_path='./modelweight/cifar_densenet121_best.pth',data_loader= dataloader, num_epochs= 12)
    result = testing(model , dataloader= dataloader)
    
    return result , train_time



if __name__ == "__main__":
    options = read_opts()  #reading hyperparameters 
    
    # Configuration of torch
    torch.set_default_dtype(torch.double)  

    #Run algorithms
    accuracy = 0 
    train_time = 0 
    if options['baseline_model']=="resnet50":
        accuracy , train_time = run_resnet50()
    elif options['baseline_model']=="resnet101":
        accuracy , train_time = run_resnet101()
    elif options['baseline_model']=='vgg16':
        accuracy , train_time = run_vgg16()
    elif options['baseline_model']=='densenet121':
        accuracy , train_time = run_densenet121()
    elif options['baseline_model']=='mobilenet_v2':
        accuracy , train_time = run_mobilenet_v2()
    elif options['baseline_model']=='efficientnet_b0':
        accuracy , train_time = run_efficientnet_b0()
    elif options['baseline_model']=='inception_v3':
        accuracy , train_time = run_inception_v3()
    elif options['baseline_model'] == 'mobilenet_v2_from_scratch':
        accuracy , train_time = run_mobilenet_v2_from_scratch()
    
    
    
    print("Writting results...", end="")
    f = open(f"./output.csv", "a")
    f.write(f"{options['baseline_model']} , {accuracy} , {train_time}\n")
    f.close()
    print("Done!")




# if __name__ == "__main__":

#     # Load the pre-trained VGG16 model
#     vgg16 = models.vgg16(pretrained=True)

#     # Modify the last layer to output 10 classes
#     vgg16.classifier[6] = nn.Linear(in_features=4096, out_features=10)

#     # Freeze all parameters
#     # for param in vgg16.parameters():
#     #     param.requires_grad = False

#     # # Unfreeze the parameters of the last layer and one other layer (for example, classifier[5])
#     # for param in vgg16.classifier[5].parameters():
#     #     param.requires_grad = True
#     # for param in vgg16.classifier[6].parameters():
#     #     param.requires_grad = True

#     vgg16 = unfreeze_layers(vgg16 , 2)

#     # Alternatively, if you want to train the last convolutional layer (for example, features[28])
#     # Uncomment the following lines to unfreeze it
#     # for param in vgg16.features[28].parameters():
#     #     param.requires_grad = True

#     # Calculate the number of trainable parameters
#     num_trainable_params = sum(p.numel() for p in vgg16.parameters() if p.requires_grad)

#     print(f"Number of trainable parameters: {num_trainable_params}")
