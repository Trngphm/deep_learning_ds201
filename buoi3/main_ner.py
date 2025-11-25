# main_phoner.py
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from tqdm import tqdm
import argparse
import numpy as np

# import các file bạn đã có (đặt cùng thư mục hoặc sửa path)
# file chứa PhoNERDataset, Vocab và collate_fn (padding labels = -100)
from vocabs.phoner import PhoNERDataset, collate_fn  # hoặc tên file bạn đang dùng
from models.bilstm import BiLSTMNER  # hoặc bilstm_phoner nếu bạn lưu tên khác
from configs.config_bilstm import ConfigBiLSTM

device = "cuda" if torch.cuda.is_available() else "cpu"


def flatten_preds_trues(preds_tensor, trues_tensor):
    """
    preds_tensor: [B, T] (tensor int)
    trues_tensor: [B, T] (tensor int, padded positions = -100)
    return two lists of ints (flattened) where trues != -100
    """
    preds = preds_tensor.cpu().tolist()
    trues = trues_tensor.cpu().tolist()

    flat_p = []
    flat_t = []
    for p_seq, t_seq in zip(preds, trues):
        for p, t in zip(p_seq, t_seq):
            if t != -100:
                flat_p.append(p)
                flat_t.append(t)
    return flat_p, flat_t


def evaluate(model, loader, id2tag):
    model.eval()
    all_p = []
    all_t = []

    with torch.no_grad():
        for batch in loader:
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)  # padded with -100
            logits = model(x)  # [B, T, C]
            preds = logits.argmax(dim=-1)  # [B, T]

            p_flat, t_flat = flatten_preds_trues(preds, y)
            all_p.extend(p_flat)
            all_t.extend(t_flat)

    # metrics (token-level)
    f1 = f1_score(all_t, all_p, average="macro", zero_division=0)
    prec = precision_score(all_t, all_p, average="macro", zero_division=0)
    rec = recall_score(all_t, all_p, average="macro", zero_division=0)

    # classification_report expects labels 0..n-1 and names in that order
    # Lấy danh sách nhãn thực sự xuất hiện trong true labels
    labels_in_data = sorted(list(set(all_t)))
    target_names = [id2tag[i] for i in labels_in_data]

    report = classification_report(
        all_t,
        all_p,
        labels=labels_in_data,
        target_names=target_names,
        zero_division=0
    )


    return f1, prec, rec, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=str, default="phoner_bilstm.pth")
    args = parser.parse_args()
    
    train_ds = PhoNERDataset(ConfigBiLSTM.train_file)
    dev_ds = PhoNERDataset(ConfigBiLSTM.dev_file, vocab=train_ds.vocab)
    test_ds = PhoNERDataset(ConfigBiLSTM.test_file, vocab=train_ds.vocab)

    train_dataloader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Lấy vocab từ train_ds
    vocab = train_ds.vocab


    print(f"Vocab size: {vocab.vocab_size} | Num labels: {vocab.n_labels}")

    model = BiLSTMNER(
        vocab_size=vocab.vocab_size,
        embedding_dim=ConfigBiLSTM.embedding_dim,
        hidden_size=ConfigBiLSTM.hidden_size,
        n_layers=ConfigBiLSTM.num_layers,
        num_classes=vocab.n_labels,   # BiLSTMNER expects num_classes (final linear)
        dropout=ConfigBiLSTM.dropout
    ).to(device)

    # loss with ignore_index = -100 (the padding value in collate_fn)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.Adam(model.parameters(), lr=ConfigBiLSTM.learning_rate)

    best_f1 = 0.0

    EPOCHS = 10  
    for epoch in range(EPOCHS):
        model.train()
        losses = []
        pbar = tqdm(train_dataloader, desc=f"Training epoch {epoch + 1}", leave=True)
        for batch in pbar:
            x = batch["input_ids"].to(device)      # [B, T]
            y = batch["labels"].to(device)         # [B, T], padded with -100

            logits = model(x)                      # [B, T, C]
            B, T, C = logits.shape

            loss = loss_fn(logits.view(B * T, C), y.view(B * T))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            pbar.set_postfix({"loss": f"{np.mean(losses):.4f}"})
            
        # In loss trung bình sau epoch
        print(f"Epoch {epoch + 1} - Loss trung bình: {np.mean(losses):.4f}")

        # evaluate on dev
        f1, prec, rec, report = evaluate(model, dev_dataloader, vocab.id2tag)
        print(f"\nDev metrics — F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")
        # print(report)  # uncomment để xem báo cáo chi tiết

        # save best
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), args.save)
            print(f"Saved best model (F1={best_f1:.4f}) -> {args.save}")

    # Final test
    print("\n=== Final test on best model ===")
    # load best model weights (already saved)
    model.load_state_dict(torch.load(args.save, map_location=device))
    f1, prec, rec, report = evaluate(model, test_dataloader, vocab.id2tag)
    print(f"Test metrics — F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")
    print("\nClassification report (token-level):")
    print(report)


if __name__ == "__main__":
    main()
