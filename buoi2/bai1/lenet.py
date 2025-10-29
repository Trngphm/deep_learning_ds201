import torch
from torch import nn
from torch.nn import functional as F 

class LeNetModel(nn.Module):
    """Some Information about LeNetModule"""
    def __init__(self):
        super().__init__()

        self.conv_1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=(5, 5),
            stride=1,
            padding=(2, 2)
        )

        self.pooling_1 = nn.AvgPool2d(
            kernel_size=(2, 2),
            stride=(2, 2),
            padding=(0, 0)
        )

        self.conv_2 =  nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=(5, 5),
            padding=0
        )

        self.pooling_2 = nn.AvgPool2d(
            stride=(2, 2),
            kernel_size=(2, 2),
            padding=(0, 0)
        )

        self.conv_3 = nn.Conv2d(
            in_channels=16,
            out_channels=120,
            kernel_size=(5, 5),
            padding=(0, 0)
        )

        self.fc = nn.Linear(
            in_features=120,
            out_features=84
        )

        self.output = nn.Linear(
            in_features=84,
            out_features=10
        )

    def forward(self, images):
        images = images.unsqueeze(1)

        features = F.sigmoid(self.conv_1(images))
        features = self.pooling_1(features)

        features = F.sigmoid(self.conv_2(features))
        features = self.pooling_2(features)

        features = F.sigmoid(self.conv_3(features))
        features = features.squeeze(-1).squeeze(-1)

        features = F.sigmoid(self.fc(features))
        logits = self.output(features)

        return logits

