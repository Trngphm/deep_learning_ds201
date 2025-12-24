from torch import nn
import torch
from models.transformer import TransformerEncoder, PositionalEncoding


class TransformerNER(nn.Module):
    """
    Transformer Encoder for sequence labeling (PhoNER)
    Output: token-level tag prediction
    """

    def __init__(self, vocab, pad_idx,
                 d_model=256, n_head=8, n_layer=3, d_ff=512, dropout=0.1):
        super().__init__()
        
        self.vocab = vocab
        # Embedding
        self.embedding = nn.Embedding(
            vocab.vocab_size,
            d_model,
            padding_idx=pad_idx
        )

        self.pe = PositionalEncoding(d_model, dropout)

        # Transformer Encoder (3 layers)
        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_head=n_head,
            n_layer=n_layer,
            d_ff=d_ff,
            dropout=dropout
        )

        # Token-level classifier
        self.classifier = nn.Linear(d_model, vocab.n_labels)

        # Loss (ignore padding = -100)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, labels=None):
        """
        input_ids: [B, T]
        labels:    [B, T] (optional)
        """

        # Padding mask: 1 = valid, 0 = pad
        mask = (input_ids != 0).unsqueeze(1).unsqueeze(2)  # [B,1,1,T]

        # Embedding + Positional Encoding
        x = self.embedding(input_ids)   # [B,T,D]
        x = self.pe(x)

        # Transformer Encoder
        x = self.encoder(x, mask)       # [B,T,D]

        # Token-level logits
        logits = self.classifier(x)     # [B,T,C]

        if labels is not None:
            B, T, C = logits.shape
            loss = self.loss_fn(
                logits.view(B * T, C),
                labels.view(B * T)
            )
            return loss, logits

        return logits
