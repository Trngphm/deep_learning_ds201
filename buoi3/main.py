import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, accuracy_score
from tqdm import tqdm
import numpy as np
import argparse
import torch.nn.utils

from vocabs.uit_vsfc import collate_fn, UITVSFCDataset, Vocab
from models.lstm import LSTMClassifier
from models.gru import GRUClassifier
from configs.config_lstm import ConfigLSTM
from configs.config_gru import ConfigGRU


device = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(model, loader, idx2label):
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for batch in loader:
            x = batch["input_ids"].to(device)
            y = batch["label"].to(device)

            out = model(x).argmax(dim=1)

            preds.extend(out.tolist())
            trues.extend(y.tolist())
            
    # === Metrics ===
    acc = accuracy_score(trues, preds)
    precision = precision_score(trues, preds, average="macro", zero_division=0)
    recall = recall_score(trues, preds, average="macro", zero_division=0)
    f1 = f1_score(trues, preds, average="macro", zero_division=0)

    report = classification_report(
        trues,
        preds,
        target_names=[idx2label[i] for i in sorted(idx2label.keys())]
    )

    return acc, precision, recall, f1, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "gru"],
                        help="Chọn model để train")
    args = parser.parse_args()

    # Chọn config và model
    if args.model == "lstm":
        Config = ConfigLSTM
        ModelClass = LSTMClassifier
    elif args.model == "gru":
        Config = ConfigGRU
        ModelClass = GRUClassifier

        
    # Load dataset
    train_ds = UITVSFCDataset(Config.train_file)
    dev_ds = UITVSFCDataset(Config.dev_file, vocab=train_ds.vocab)
    test_ds = UITVSFCDataset(Config.test_file, vocab=train_ds.vocab)

    train_dataloader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    vocab = train_ds.vocab

    model = ModelClass(
        vocab_size=vocab.vocab_size,
        embedding_dim=Config.embedding_dim,
        hidden_size=Config.hidden_size,
        n_layers=Config.num_layers,
        num_classes=vocab.n_labels
    ).to(device)

    # ---- 3. Loss + Optimizer ----
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    
    # ---- 4. Train loop ----
    
    EPOCHS = 10
    for epoch in range(EPOCHS):
        print(f"\n===== Epoch {epoch + 1}/{EPOCHS} =====")
        model.train()
        losses = []

        progress_bar = tqdm(train_dataloader, desc=f"Training epoch {epoch + 1}", leave=True)
        for batch in progress_bar:
            x = batch["input_ids"].to(device)
            label = batch["label"].to(device)

            output = model(x)
            loss = loss_fn(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            progress_bar.set_postfix({"loss": f"{np.mean(losses):.4f}"})
            
        # In loss trung bình sau epoch
        print(f"Epoch {epoch + 1} - Loss trung bình: {np.mean(losses):.4f}")

        acc, precision, recall, f1, _ = evaluate(model, dev_dataloader, vocab.id2label)

        print(f"Eval - Acc: {acc:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f} | F1: {f1:.4f}")

    
    # Lưu model
    model_file = f"{args.model}_model.pth"
    torch.save(model.state_dict(), model_file)
    print(f"Model đã lưu vào: {model_file}")
    
    # =============== FINAL FULL REPORT ===============
    print("\n=== Final Classification Report ===")
    _, _, _, _, report = evaluate(model, test_dataloader, vocab.id2label)
    print(report)



if __name__ == "__main__":
    main()
