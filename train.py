import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model.alexnet import AlexNet
from utils.trainer import train_model, evaluate_model

if __name__ == '__main__':
    
    
    transforms = transforms.Compose([
        transforms.Resize([227, 227]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=False, num_workers=2)

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AlexNet(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)

    train_model(model, trainloader, criterion, optimizer, device, num_epochs=10)    
    evaluate_model(model, testloader, device)

            
