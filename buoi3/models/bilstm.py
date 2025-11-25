import torch
import torch.nn as nn

class BiLSTMNER(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size,
                 n_layers, num_classes, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0
        )

        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(dropout)

        # Bidirectional → hidden_size * 2
        self.fc = nn.Linear(hidden_size * 2, num_classes)

        self.init_weights()

    def init_weights(self):
        """Khởi tạo trọng số chuẩn cho LSTM"""
        for name, param in self.bilstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, input_ids):
        """
        input_ids: [batch, seq_len]
        output:    [batch, seq_len, num_classes]
        """

        mask = (input_ids != 0).float().unsqueeze(-1)  # 0 = pad

        x = self.embedding(input_ids)
        x = self.dropout(x)

        x = x * mask  # mask embedding

        lstm_out, _ = self.bilstm(x)
        lstm_out = self.dropout(lstm_out)

        logits = self.fc(lstm_out)  # [B, L, C]

        return logits
