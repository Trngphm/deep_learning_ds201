import torch

class ConfigGRU:
    # Data paths
    data_path = "data/uit-vsfc/"
    train_file = "./data/UIT-VSFC/UIT-VSFC-train.json"
    dev_file = "./data/UIT-VSFC/UIT-VSFC-dev.json"
    test_file = "./data/UIT-VSFC/UIT-VSFC-test.json"
    
    # Model architecture
    hidden_size = 256
    num_layers = 5
    dropout = 0.3
    embedding_dim = 300
    num_classes = 3  # Positive, Negative, Neutral, etc.
    
    # Training parameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 50
    patience = 5
    
    # Vocabulary
    max_length = 256
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)