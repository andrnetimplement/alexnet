import torch
import torch.nn as nn
import torch.functional as F

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000) -> None:
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        
        # 227 x 227 x 3 * 11 x 11 x 3
        self.feautures = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
            )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(6 * 6 * 256, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.num_classes),
        )
    
    def forward(self, x):
        
        x = self.feautures(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x
    
    
    