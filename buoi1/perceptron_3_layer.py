import torch
from torch import nn
from torch.nn import functional as F

class Perceptron_3_layer(nn.Module):
    def __init__(self, image_size, num_labels, hidden1=256, hidden2=128):
        super().__init__()
        w, h = image_size
        input_dim = w * h

        # Layer 1: input -> hidden1 + ReLU
        self.fc1 = nn.Linear(input_dim, hidden1)
        # Layer 2: hidden1 -> hidden2 + ReLU
        self.fc2 = nn.Linear(hidden1, hidden2)
        # Layer 3: hidden2 -> output + Softmax
        self.fc3 = nn.Linear(hidden2, num_labels)

    def forward(self, x):
        bs = x.shape[0]
        x = x.reshape(bs, -1)          
        x = F.relu(self.fc1(x))        
        x = F.relu(self.fc2(x))        
        x = F.log_softmax(self.fc3(x), dim=-1) 
        return x
