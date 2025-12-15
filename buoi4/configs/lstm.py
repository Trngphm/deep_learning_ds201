import torch

class ConfigLSTM:
    # Data paths
    data_path = "data/uit-vsfc/"
    train_file = "./data/small-train.json"
    dev_file = "./data/small-dev.json"
    test_file = "./data/small-test.json"
    
    # Model architecture
    hidden_size = 256
    num_layers = 3
    dropout = 0.3
    embed_size = 300
    
    # Training parameters
    batch_size = 32
    lr = 5e-4
    num_epochs = 10
    patience = 5
    
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)