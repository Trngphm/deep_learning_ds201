import torch
import torch.nn as nn
import torch.nn.functional as F
from vocabs.vocab import Vocab

# -----------------------------
# Bahdanau Attention
# -----------------------------
class BahdanauAttention(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size, attn_dim=None):
        """
        enc_hidden_size: hidden_size * 2 (because encoder is bidirectional)
        dec_hidden_size: hidden_size (decoder)
        """
        super().__init__()
        if attn_dim is None:
            attn_dim = dec_hidden_size  # common choice
        self.W_enc = nn.Linear(enc_hidden_size, attn_dim, bias=False)  # apply to encoder outputs
        self.W_dec = nn.Linear(dec_hidden_size, attn_dim, bias=False)  # apply to decoder hidden
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, dec_hidden, enc_outputs, enc_mask):
        """
        dec_hidden: (batch, dec_hidden)  -- last layer hidden of decoder (not num_layers dim)
        enc_outputs: (batch, src_len, enc_hidden_size)  -- batch_first
        enc_mask: (batch, src_len)  -- 1 for valid tokens, 0 for pad
        returns:
            context: (batch, enc_hidden_size)
            attn_weights: (batch, src_len)
        """
        # enc_outputs -> (batch, src_len, attn_dim)
        enc_proj = self.W_enc(enc_outputs)              # (batch, src_len, attn_dim)
        dec_proj = self.W_dec(dec_hidden).unsqueeze(1)  # (batch, 1, attn_dim)

        energy = torch.tanh(enc_proj + dec_proj)        # (batch, src_len, attn_dim)
        scores = self.v(energy).squeeze(-1)             # (batch, src_len)

        # mask out pads
        if enc_mask is not None:
            scores = scores.masked_fill(enc_mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=1)        # (batch, src_len)
        context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs).squeeze(1)  # (batch, enc_hidden_size)

        return context, attn_weights

# -----------------------------
# Encoder (5-layer LSTM, bidirectional)
# -----------------------------
class EncoderWithAttention(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers=5, dropout=0.2):
        """
        hidden_size: hidden size per direction (so outputs' last dim = hidden_size * 2)
        """
        super().__init__()
        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

    def forward(self, src):
        """
        src: (batch, src_len)
        returns:
            outputs: (batch, src_len, hidden_size*2)
            (h_n, c_n): each (num_layers*2, batch, hidden_size)
        """
        embedded = self.embedding(src)  # (batch, src_len, embed_size)
        outputs, (h_n, c_n) = self.lstm(embedded)  # outputs: (batch, src_len, 2*hidden)
        return outputs, (h_n, c_n)

# -----------------------------
# Decoder with Attention (5-layer LSTM)
# -----------------------------
class DecoderWithAttention(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, enc_hidden_size, num_layers=5, dropout=0.2):
        """
        enc_hidden_size: hidden_size * 2 (encoder is bidirectional)
        hidden_size: decoder hidden_size (per layer)
        """
        super().__init__()
        self.output_size = output_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.enc_hidden_size = enc_hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(output_size, embed_size)
        self.attention = BahdanauAttention(enc_hidden_size, hidden_size)

        # input to LSTM at each step: embed + context(enc_hidden_size)
        self.lstm = nn.LSTM(
            embed_size + enc_hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc_out = nn.Linear(hidden_size + enc_hidden_size + embed_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, cell, enc_outputs, enc_mask):
        """
        input_token: (batch,) token indices for current step
        hidden: (num_layers, batch, hidden_size)
        cell:   (num_layers, batch, hidden_size)
        enc_outputs: (batch, src_len, enc_hidden_size)
        enc_mask: (batch, src_len)
        returns:
            prediction: (batch, output_size)
            hidden, cell: updated LSTM states
            attn_weights: (batch, src_len)
        """
        batch = input_token.size(0)
        # embed
        embed = self.embedding(input_token).unsqueeze(1)  # (batch, 1, embed_size)
        embed = self.dropout(embed)

        # use top layer hidden for attention (hidden[-1] -> (batch, hidden_size))
        dec_top_hidden = hidden[-1]  # (batch, hidden_size)
        context, attn_weights = self.attention(dec_top_hidden, enc_outputs, enc_mask)  # context: (batch, enc_hidden_size)
        context = context.unsqueeze(1)  # (batch, 1, enc_hidden_size)

        # prepare LSTM input
        lstm_input = torch.cat((embed, context), dim=2)  # (batch, 1, embed + enc_hidden)

        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))  # output: (batch,1,hidden_size)

        output = output.squeeze(1)   # (batch, hidden_size)
        context = context.squeeze(1) # (batch, enc_hidden_size)
        embed = embed.squeeze(1)     # (batch, embed_size)

        # concat for final prediction
        pred_input = torch.cat((output, context, embed), dim=1)  # (batch, hidden + enc_hidden + embed)
        prediction = self.fc_out(pred_input)                     # (batch, output_size)

        return prediction, hidden, cell, attn_weights

# -----------------------------
# Seq2Seq wrapper (training + predict)
# -----------------------------
class LSTMAttn(nn.Module):
    def __init__(self, encoder: EncoderWithAttention, decoder: DecoderWithAttention, vocab: Vocab, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab = vocab
        self.device = device
        
        self.MAX_LEN = vocab.max_sentence_length + 2  # +2 for <sos> and <eos>

        # projection to convert encoder (bi) states to decoder initial states
        enc_h_dim = encoder.hidden_size * 2
        dec_h_dim = decoder.hidden_size
        num_layers = encoder.num_layers

        # Project concatenated (forward||backward) per layer to decoder hidden_size
        # We'll apply these to tensors with shape (num_layers, batch, 2*hidden)
        self.enc2dec_h = nn.Linear(enc_h_dim, dec_h_dim)
        self.enc2dec_c = nn.Linear(enc_h_dim, dec_h_dim)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    def _init_decoder_state_from_encoder(self, h_n, c_n):
        """
        h_n, c_n: (num_layers*2, batch, enc_hidden_per_dir)
        We reshape -> (num_layers, 2, batch, enc_hidden) and concat forward+backward for each layer:
            => (num_layers, batch, enc_hidden*2)
        Then project with enc2dec_* to (num_layers, batch, dec_hidden)
        Return as (num_layers, batch, dec_hidden)
        """
        num_layers_times_2, batch, enc_h = h_n.size()
        num_layers = num_layers_times_2 // 2
        enc_hidden = enc_h  # per-dir hidden size

        # reshape
        h_n = h_n.view(num_layers, 2, batch, enc_hidden)  # (num_layers, 2, batch, enc_hidden)
        c_n = c_n.view(num_layers, 2, batch, enc_hidden)

        # concat forward and backward for each layer along last dim
        h_cat = torch.cat([h_n[:,0,:,:], h_n[:,1,:,:]], dim=2)  # (num_layers, batch, enc_hidden*2)
        c_cat = torch.cat([c_n[:,0,:,:], c_n[:,1,:,:]], dim=2)

        # project to decoder hidden size
        # apply linear on last dim: convert (num_layers, batch, 2*enc_hidden) -> (num_layers, batch, dec_hidden)
        # linear expects (N, in_features) so we reshape
        nl, b, feat = h_cat.size()
        h_cat_flat = h_cat.view(-1, feat)  # (num_layers*batch, feat)
        c_cat_flat = c_cat.view(-1, feat)

        h_proj = torch.tanh(self.enc2dec_h(h_cat_flat)).view(nl, b, -1)
        c_proj = torch.tanh(self.enc2dec_c(c_cat_flat)).view(nl, b, -1)

        return h_proj, c_proj  # each (num_layers, batch, dec_hidden)

    def forward(self, src, tgt, teacher_forcing_ratio=1.0):
        """
        src: (batch, src_len)
        tgt: (batch, tgt_len)
        src_mask: (batch, src_len) 1=valid, 0=pad
        """
        self.train()
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        vocab_size = self.vocab.total_tgt_tokens
        
        src_mask = (src != self.vocab.pad_idx).long()

        # Encode
        enc_outputs, (h_n, c_n) = self.encoder(src)  # enc_outputs: (batch, src_len, enc_hidden*2)

        # init decoder hidden/cell from encoder
        dec_h, dec_c = self._init_decoder_state_from_encoder(h_n, c_n)  # (num_layers, batch, dec_hidden)

        # first input token = <sos>
        input_token = tgt[:, 0]  # (batch,)

        logits = torch.zeros(batch_size, tgt_len, vocab_size, device=self.device)

        for t in range(1, tgt_len):
            output, dec_h, dec_c, _ = self.decoder(input_token, dec_h, dec_c, enc_outputs, src_mask)
            logits[:, t, :] = output

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = tgt[:, t] if teacher_force else top1

        loss = self.loss_fn(logits[:, 1:].reshape(-1, vocab_size), tgt[:, 1:].reshape(-1))
        return loss

    @torch.no_grad()
    def predict(self, src):
        self.eval()
        batch = src.size(0)
        src_mask = (src != self.vocab.pad_idx).long()

        enc_outputs, (h_n, c_n) = self.encoder(src)
        dec_h, dec_c = self._init_decoder_state_from_encoder(h_n, c_n)

        input_token = torch.tensor([self.vocab.bos_idx] * batch, device=self.device)
        outputs = [[] for _ in range(batch)]

        for _ in range(self.MAX_LEN):
            output, dec_h, dec_c, attn = self.decoder(input_token, dec_h, dec_c, enc_outputs, src_mask)
            top1 = output.argmax(1)  # (batch,)

            for i in range(batch):
                if top1[i].item() == self.vocab.eos_idx:
                    continue
                outputs[i].append(top1[i].item())

            input_token = top1

        # convert idx -> words
        decoded = []
        for seq in outputs:
            decoded.append([self.vocab.tgt_itos[idx] for idx in seq])
        return decoded
