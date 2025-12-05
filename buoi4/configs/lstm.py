import torch

class ConfigLSTM:
    # Data paths
    data_path = "data/uit-vsfc/"
    train_file = "./data/train_small.json"
    dev_file = "./data/dev_small.json"
    test_file = "./data/test_small.json"
    
    # Model architecture
    hidden_size = 256
    num_layers = 5
    dropout = 0.3
    embed_size = 300
    
    # Training parameters
    batch_size = 32
    lr = 5e-4
    num_epochs = 50
    patience = 5
    
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)