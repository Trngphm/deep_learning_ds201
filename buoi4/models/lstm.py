import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from vocabs.vocab import Vocab


class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers=5, dropout=0.2):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, 
            batch_first=True, dropout=dropout, bidirectional=False
        )
        
    def forward(self, x):
        # x: [batch, seq_len]
        embedded = self.embedding(x)  # [batch, seq_len, embed_size]
        outputs, (h_n, c_n) = self.lstm(embedded)
        # outputs: [batch, seq_len, hidden_size]
        # h_n, c_n: [num_layers, batch, hidden_size]
        return outputs, (h_n, c_n)
    
class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, num_layers=5, dropout=0.2):
        super(Decoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(output_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, 
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden):
        # x: [batch] → token index
        x = x.unsqueeze(1)  # [batch, 1]
        embedded = self.embedding(x)  # [batch, 1, embed_size]
        output, hidden = self.lstm(embedded, hidden)  # output: [batch, 1, hidden]
        output = self.fc(output.squeeze(1))  # [batch, output_size]
        return output, hidden

class LSTM(nn.Module):
    def __init__(self, encoder, decoder, vocab: Vocab, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        self.vocab = vocab
        
        self.MAX_LEN = vocab.max_sentence_length + 2  # +2 for <sos> and <eos>
        
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    def forward(self, src, tgt, teacher_forcing_ratio=1.0):
        """
        src: [batch, src_len]
        tgt: [batch, tgt_len]
        """
        self.train()
        
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        vocab_size = self.vocab.total_tgt_tokens

        # ---- ENCODER ----
        encoder_outputs, (h, c) = self.encoder(src)

        # Decoder input đầu tiên = <sos>
        input_token = tgt[:, 0]    # [batch]

        logits = torch.zeros(batch_size, tgt_len, vocab_size, device=self.device)

        hidden = (h, c)

        for t in range(1, tgt_len):
            # ---- DECODER ----
            output, hidden = self.decoder(input_token, hidden)
            logits[:, t, :] = output

            # Teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)

            input_token = tgt[:, t] if teacher_force else top1

        loss = self.loss_fn(logits[:, 1:].reshape(-1, vocab_size), tgt[:, 1:].reshape(-1))
        
        return loss

    @torch.no_grad()
    def predict(self, src):
        self.eval()
        
        batch_size = src.size(0)
        encoder_outputs, (h, c) = self.encoder(src)

        input_token = torch.tensor([self.vocab.sos_idx] * batch_size,
                                   device=self.device)

        hidden = (h, c)

        outputs = [[] for _ in range(batch_size)]

        for _ in range(self.MAX_LEN):
            output, hidden = self.decoder(input_token, hidden)
            top1 = output.argmax(1)

            for b in range(batch_size):
                if top1[b].item() == self.vocab.eos_idx:
                    continue
                outputs[b].append(top1[b].item())

            input_token = top1

        # convert to tokens
        decoded = []
        for seq in outputs:
            tokens = [self.vocab.idx2word[i] for i in seq]
            decoded.append(tokens)

        return decoded
