import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
import numpy as np
import argparse

from vocabs.dataset import PhoMTDataset, collate_fn
from vocabs.vocab import Vocab
from metrics import compute_metrics

from models.lstm import Encoder, Decoder, LSTM
from models.bai2 import EncoderWithAttention, DecoderWithAttention, LSTMAttn
from models.bai3 import EncoderLuong, DecoderLuong, LSTMLuong
from configs.lstm import ConfigLSTM 

device = "cuda" if torch.cuda.is_available() else "cpu"


# ======================= EVALUATION ===========================
@torch.no_grad()
def evaluate(model, loader, vocab):
    model.eval()

    all_preds = []
    all_refs = []
    
    progress_bar = tqdm(loader, desc="Evaluating")

    for batch in progress_bar:
        src = batch["vietnamese"].to(device)
        tgt = batch["english"].to(device)

        # model.predict → list[list[str]]
        preds = model.predict(src)

        # ground truth
        batch_refs = []
        for sent in tgt:
            seq = []
            for idx in sent[1:]:  # bỏ <bos>
                idx = idx.item()
                if idx in (vocab.pad_idx, vocab.eos_idx):
                    break
                seq.append(vocab.tgt_itos[idx])
            batch_refs.append(" ".join(seq))


        # preds là list[list[token]] → convert sang string
        preds = [" ".join(p) for p in preds]

        all_preds.extend(preds)
        all_refs.extend(batch_refs)

    metrics = compute_metrics(all_preds, all_refs)

    return metrics


# ======================= MAIN TRAIN LOOP ======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lstm",
                        choices=["lstm", "lstm_attn", "lstm_luong"],
                        help="Model dịch máy")
    args = parser.parse_args()

    # -------- Load Dataset ----------
    vocab = Vocab(path="data", src_lang="vietnamese", tgt_lang="english")
    vocab.make_vocab("data", "vietnamese", "english")
    
    train_ds = PhoMTDataset(ConfigLSTM.train_file, vocab)
    dev_ds   = PhoMTDataset(ConfigLSTM.dev_file, vocab)
    test_ds  = PhoMTDataset(ConfigLSTM.test_file, vocab)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    dev_loader   = DataLoader(dev_ds,   batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, collate_fn=collate_fn)

    # =========== chọn model theo args.model ===========
    if args.model == "lstm":
        Config = ConfigLSTM

        encoder = Encoder(
            input_size=vocab.total_src_tokens,
            embed_size=Config.embed_size,
            hidden_size=Config.hidden_size,
            num_layers=5,
            dropout=0.2
        )
        decoder = Decoder(
            output_size=vocab.total_tgt_tokens,
            embed_size=Config.embed_size,
            hidden_size=Config.hidden_size,
            num_layers=5,
            dropout=0.2
        )
        model = LSTM(encoder, decoder, vocab, device).to(device)

    elif args.model == "lstm_attn":
        Config = ConfigLSTM

        encoder = EncoderLuong(
            input_size=vocab.total_src_tokens,
            embed_size=Config.embed_size,
            hidden_size=Config.hidden_size,
            num_layers=5,
            dropout=0.2
        )
        decoder = DecoderLuong(
            output_size=vocab.total_tgt_tokens,
            embed_size=Config.embed_size,
            hidden_size=Config.hidden_size,
            enc_hidden_size=Config.hidden_size * 2,
            num_layers=5,
            dropout=0.2
        )
        model = LSTMLuong(encoder, decoder, vocab, device).to(device)

    elif args.model == "lstm_luong":
        Config = ConfigLSTM

        encoder = EncoderLuong(
            input_size=vocab.total_src_tokens,
            embed_size=Config.embed_size,
            hidden_size=Config.hidden_size,
            num_layers=5,
            dropout=0.2
        )
        decoder = DecoderLuong(
            output_size=vocab.total_tgt_tokens,
            embed_size=Config.embed_size,
            hidden_size=Config.hidden_size,
            num_layers=5,
            dropout=0.2
        )
        model = LSTMLuong(encoder, decoder, vocab, device).to(device)

    # -------- loss + optimizer ----------
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)

    # -------- training ----------
    EPOCHS = Config.num_epochs

    for epoch in range(EPOCHS):
        print(f"\n========== Epoch {epoch+1}/{EPOCHS} ==========")
        model.train()
        losses = []

        progress_bar = tqdm(train_loader, desc=f"Training epoch {epoch+1}")

        for batch in progress_bar:
            src = batch["vietnamese"].to(device)
            tgt = batch["english"].to(device)

            loss = model(src, tgt, teacher_forcing_ratio=1.0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            losses.append(loss.item())
            progress_bar.set_postfix({"loss": f"{np.mean(losses):.4f}"})

        print(f"Epoch {epoch+1} - Avg Loss: {np.mean(losses):.4f}")

        # ---- evaluate dev ----
        metrics = evaluate(model, dev_loader, vocab)

        print(f"[Dev] BLEU1:  {metrics['BLEU@1']:.4f}")
        print(f"[Dev] BLEU2:  {metrics['BLEU@2']:.4f}")
        print(f"[Dev] BLEU3:  {metrics['BLEU@3']:.4f}")
        print(f"[Dev] BLEU4:  {metrics['BLEU@4']:.4f}")
        print(f"[Dev] ROUGE-L:{metrics['ROUGE-L']:.4f}")
        print(f"[Dev] METEOR: {metrics['METEOR']:.4f}")
        

    # ========== SAVE MODEL ==========
    save_path = f"{args.model}_phomt_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Đã lưu model vào: {save_path}")

    # ========== FINAL TEST REPORT ==========
    print("\n========== TEST SET REPORT ==========")
    metrics = evaluate(model, dev_loader, vocab)

    print(f"[Dev] BLEU1:  {metrics['BLEU@1']:.4f}")
    print(f"[Dev] BLEU2:  {metrics['BLEU@2']:.4f}")
    print(f"[Dev] BLEU3:  {metrics['BLEU@3']:.4f}")
    print(f"[Dev] BLEU4:  {metrics['BLEU@4']:.4f}")
    print(f"[Dev] ROUGE-L:{metrics['ROUGE-L']:.4f}")
    print(f"[Dev] METEOR: {metrics['METEOR']:.4f}")



if __name__ == "__main__":
    main()
