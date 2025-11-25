import torch

class ConfigBiLSTM:
    # Data paths
    train_file = "./data/phoner/train_word_fixed.json"
    dev_file = "./data/phoner/dev_word_fixed.json"
    test_file = "./data/phoner/test_word_fixed.json"
    
    # Model architecture
    hidden_size = 256
    num_layers = 5
    dropout = 0.3
    embedding_dim = 300
    
    # Training parameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 50
    patience = 5
    
    # Vocabulary
    max_vocab_size = 50000
    max_length = 256
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)