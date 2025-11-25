import torch 
import torch.nn as nn
import torch.nn.functional as F

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, n_layers, num_classes, dropout=0.1):
        super(LSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = n_layers
        self.num_classes = num_classes
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=False
        )
        
        #dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)
        
        self.__init_weights()
        
    def __init_weights(self):
        """Initialize weights for better convergence"""
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)
        
    def forward(self, input_ids, attention_mask=None):
        embedded = self.dropout(self.embedding(input_ids))
        
        # attention mask
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            embedded = embedded * mask
            
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        hidden = hidden[-1]  # Lấy hidden state của layer cuối cùng
        
        # output
        output = self.dropout(hidden)
        logits = self.fc(output)
        
        return logits