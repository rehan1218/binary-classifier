import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torchvision.models as models


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super(MnistModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class ResNet18(BaseModel):
    def __init__(self, num_classes=8):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        '''
        ct = 0
        for child in self.model.children():
            #print("child ", ct, ": \n", child)
            if ct<7:
                for param in child.parameters():
                    param.requires_grad = False
            ct += 1
        '''
        
    def forward(self, x):
        output = self.model(x)
        return output