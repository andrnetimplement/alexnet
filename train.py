import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
from model.alexnet import AlexNet

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

    def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            start_time = time.time()
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(labels)
                
                optimizer.zero_grad()
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
            end_time = time.time()
            
            epoch_loss = running_loss / len(train_loader)
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Time: {end_time - start_time:.2f}s')
        
        print('Finished Training')

    # Train the model
    train_model(model, trainloader, criterion, optimizer, num_epochs=10)

    def evaluate_model(model, test_loader):
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        accuracy = 100 * correct / total
        print(f'Accuracy of the model on the test images: {accuracy:.2f}%')
        
    evaluate_model(model, testloader)

            
