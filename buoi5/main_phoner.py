# main_checkpoints/phoner_transformer.py
import torch
from torch.utils.data import DataLoader
from torch import optim
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from tqdm import tqdm
import numpy as np
import os

from vocab.phoner import PhoNERDataset, collate_fn
from models.phoner import TransformerNER
from config.phoner import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, SYSTEM_CONFIG

# =====================
# DEVICE & SEED
# =====================
device = SYSTEM_CONFIG["device"]
if device == "cuda" and not torch.cuda.is_available():
    device = "cpu"

torch.manual_seed(SYSTEM_CONFIG["seed"])


# =====================
# EVALUATION 
# =====================
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


# =====================
# MAIN
# =====================
def main():
    # ===== CHECKPOINT FOLDER =====
    dataset_name = DATA_CONFIG.get("dataset_name", "phoner")
    save_dir = os.path.join("checkpoints", dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "transformer.pth")

    # ===== DATA =====
    train_ds = PhoNERDataset(DATA_CONFIG["train_file"])
    vocab = train_ds.vocab

    dev_ds = PhoNERDataset(DATA_CONFIG["dev_file"], vocab=vocab)
    test_ds = PhoNERDataset(DATA_CONFIG["test_file"], vocab=vocab)

    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=True,
        collate_fn=collate_fn
    )

    dev_loader = DataLoader(
        dev_ds,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=False,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=False,
        collate_fn=collate_fn
    )

    print("Vocab size:", vocab.vocab_size)
    print("Num NER tags:", vocab.n_labels)

    # ===== MODEL =====
    model = TransformerNER(
        vocab=vocab,
        pad_idx=0,
        d_model=MODEL_CONFIG["d_model"],
        n_head=MODEL_CONFIG["n_head"],
        n_layer=MODEL_CONFIG["n_layer"],
        d_ff=MODEL_CONFIG["d_ff"],
        dropout=MODEL_CONFIG["dropout"]
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=TRAIN_CONFIG["learning_rate"]
    )

    # ===== TRAIN =====
    best_f1 = 0.0

    for epoch in range(TRAIN_CONFIG["epochs"]):
        model.train()
        losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)

            loss, _ = model(x, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            pbar.set_postfix(loss=f"{np.mean(losses):.4f}")

        f1, p, r, _ = evaluate(model, dev_loader, vocab.id2tag)
        print(f"Dev — F1: {f1:.4f} | Precision: {p:.4f} | Recall: {r:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model (F1={best_f1:.4f})")

    # ===== TEST =====
    print("\n=== TEST ===")
    model.load_state_dict(torch.load(save_path, map_location=device))
    f1, p, r, report = evaluate(model, test_loader, vocab.id2tag)
    print(f"Test — F1: {f1:.4f} | Precision: {p:.4f} | Recall: {r:.4f}")
    print("\nClassification Report:\n")
    print(report)

if __name__ == "__main__":
    main()
