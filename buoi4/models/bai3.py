import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from vocabs.vocab import Vocab

import torch
import torch.nn as nn
import torch.nn.functional as F


class LuongAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        """
        decoder_hidden: [batch, hidden]
        encoder_outputs: [batch, src_len, hidden]
        """
        # (batch, hidden) → (batch, 1, hidden)
        dec = decoder_hidden.unsqueeze(1)

        # general score = h_t^T * Wa * h_s
        scores = torch.bmm(dec, self.Wa(encoder_outputs).transpose(1, 2))
        # (batch, 1, src_len)

        attn_weights = torch.softmax(scores, dim=-1)     # normalize
        context = torch.bmm(attn_weights, encoder_outputs)
        # context: (batch, 1, hidden)

        return context, attn_weights

class EncoderLuong(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers=5, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (h, c) = self.lstm(embedded)
        # outputs: (batch, src_len, hidden)
        return outputs, (h, c)
    
class DecoderLuong(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, num_layers=5, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(output_size, embed_size)
        
        self.lstm = nn.LSTM(
            embed_size + hidden_size,     # concat(embedding, context)
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.attn = LuongAttention(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, token, hidden, encoder_outputs):
        """
        token: [batch]
        hidden: (h, c)
        encoder_outputs: [batch, src_len, hidden]
        """
        embedded = self.embedding(token).unsqueeze(1)  # (batch, 1, embed)

        # attention
        dec_hidden_top = hidden[0][-1]     # top layer, shape [batch, hidden]
        context, attn_weights = self.attn(dec_hidden_top, encoder_outputs)

        # concat embedding + context
        rnn_input = torch.cat([embedded, context], dim=2)

        output, hidden = self.lstm(rnn_input, hidden)  # (batch, 1, hidden)

        output = output.squeeze(1)
        context = context.squeeze(1)

        logits = self.fc(torch.cat([output, context], dim=1))  # (batch, vocab)

        return logits, hidden, attn_weights


class LSTMLuong(nn.Module):
    def __init__(self, encoder, decoder, vocab: Vocab, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab = vocab
        self.device = device

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
        self.MAX_LEN = vocab.max_sentence_length + 2

    def forward(self, src, tgt, teacher_forcing_ratio=1.0):
        batch = src.size(0)
        tgt_len = tgt.size(1)
        vocab = self.vocab.total_tgt_tokens

        encoder_outputs, hidden = self.encoder(src)

        input_token = tgt[:, 0]     # <sos>
        logits = torch.zeros(batch, tgt_len, vocab, device=self.device)

        for t in range(1, tgt_len):
            output, hidden, _ = self.decoder(input_token, hidden, encoder_outputs)
            logits[:, t, :] = output

            teacher = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = tgt[:, t] if teacher else top1

        loss = self.loss_fn(
            logits[:, 1:].reshape(-1, vocab),
            tgt[:, 1:].reshape(-1)
        )
        return loss

    @torch.no_grad()
    def predict(self, src):
        self.eval()

        encoder_outputs, hidden = self.encoder(src)

        batch = src.size(0)
        input_token = torch.tensor([self.vocab.bos_idx] * batch, device=self.device)

        outputs = [[] for _ in range(batch)]

        for _ in range(self.MAX_LEN):
            output, hidden, _ = self.decoder(input_token, hidden, encoder_outputs)
            top1 = output.argmax(1)

            for b in range(batch):
                if top1[b].item() == self.vocab.eos_idx:
                    continue
                outputs[b].append(top1[b].item())

            input_token = top1

        decoded = []
        for seq in outputs:
            decoded.append([self.vocab.tgt_itos[i] for i in seq])

        return decoded
