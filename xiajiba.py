import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexSAPblock(nn.Module):
    def __init__(self, in_channels):
        super(ComplexSAPblock, self).__init__()

        # Branch 1
        self.conv3x3_1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)

        # Branch 2
        self.conv3x3_2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

        # Branch 3
        self.conv5x5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(in_channels)

        
        self.fc1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Branch 1
        branch1 = self.conv3x3_1(x)
        branch1 = self.bn1(branch1)

        # Branch 2
        branch2 = self.conv3x3_2(x)
        branch2 = self.bn2(branch2)

        # Branch 3
        branch3 = self.conv5x5(x)
        branch3 = self.bn3(branch3)

        
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        avg_pool = self.fc1(avg_pool)
        avg_pool = self.fc2(avg_pool)
        channel_att = torch.sigmoid(avg_pool)

        # Multi-scale feature fusion
        feat = branch1 + branch2 + branch3
        feat = feat * channel_att

        ax = self.relu(self.gamma * feat + (1 - self.gamma) * x)

        return ax
