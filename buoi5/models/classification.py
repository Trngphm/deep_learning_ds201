from torch import nn
import torch
import torch.nn.functional as F
from models.transformer import TransformerEncoder, PositionalEncoding

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, num_labels, pad_idx,
                 d_model=256, n_head=8, n_layer=3, d_ff=512, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pe = PositionalEncoding(d_model, dropout)
        
        self.encoder = TransformerEncoder(
            d_model, n_head, n_layer, d_ff, dropout
        )
        
        self.classifier = nn.Linear(d_model, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, labels=None):
        mask = (input_ids != 0).unsqueeze(1).unsqueeze(2)
        
        x = self.embedding(input_ids)
        x = self.pe(x)
        x = self.encoder(x, mask)
        
        # Mean Pooling
        x = x.mean(dim=1)
        
        logits = self.classifier(x)
        
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, logits
        
        return logits
