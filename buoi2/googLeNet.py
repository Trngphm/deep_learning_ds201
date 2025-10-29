import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # left 
        self.conv_1_1 = nn.Conv2d(channels, channels, kernel_size=1)

        # middle
        self.reduce_3 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_3_3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        self.reduce_5 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_5_5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2)

        # right
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool_proj = nn.Conv2d(channels, channels, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b1 = self.relu(self.conv_1_1(x))

        r3 = self.relu(self.reduce_3(x))
        b2 = self.relu(self.conv_3_3(r3))

        r5 = self.relu(self.reduce_5(x))
        b3 = self.relu(self.conv_5_5(r5))

        b4 = self.relu(self.pool_proj(self.pool(x)))

        out = torch.cat([b1, b2, b3, b4], dim=1)
        return out

class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Stem
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        # Inception modules
        self.inception3 = InceptionBlock(128)
        self.inception4 = InceptionBlock(128 * 4)  # vì mỗi Inception nhân 4 số kênh
        self.inception5 = InceptionBlock(128 * 16)

        # Global average pooling + FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(128 * 64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.inception3(x)
        x = self.inception4(x)
        x = self.inception5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
