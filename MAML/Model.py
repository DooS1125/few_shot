import torch
from torch import nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, Nway):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
                        nn.Conv2d(1, 4, 2), nn.ReLU(),
                        nn.MaxPool2d(2, 2),
                        nn.Conv2d(4, 8, 2, stride=1, padding=1), nn.ReLU(),
                        nn.MaxPool2d(2, 2),
                        nn.Conv2d(8, 16, 2, stride=1, padding=1), nn.ReLU(),
                        nn.Conv2d(16, 16, 2, stride=1), nn.ReLU(),
                        nn.Conv2d(16, 16, 2, stride=1), nn.ReLU())
 
        self.classifier = nn.Sequential(nn.Linear(16, 8) , nn.ReLU(), nn.Linear(8, Nway)) 
        
    def forward(self, x): # 모델 연산의 순서를 정의
        x = self.features(x)
        x = torch.mean(x, [2,3])
        x = self.classifier(x)
        return x

    def parameterised(self, x, weights):
        x = F.relu(F.conv2d(x, weight=weights[0], bias=weights[1], stride=1))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(F.conv2d(x, weight=weights[2], bias=weights[3], stride=1, padding=1))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(F.conv2d(x, weight=weights[4], bias=weights[5], stride=1, padding=1))
        x = F.relu(F.conv2d(x, weight=weights[6], bias=weights[7], stride=1))
        x = F.relu(F.conv2d(x, weight=weights[8], bias=weights[9], stride=1))
        x = torch.mean(x, [2,3])
        x = F.relu(F.linear(x, weights[10], weights[11]))
        x = F.linear(x, weights[12], weights[13])
        return x 