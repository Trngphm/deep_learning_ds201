# main_checkpoints/phoner_transformer.py
import torch
from torch.utils.data import DataLoader
from torch import optim
from sklearn.metrics import f1_score, precision_score, recall_score
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
# EVALUATION (token-level)
# =====================
def evaluate(model, loader):
    model.eval()
    all_p, all_t = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)

            logits = model(x)
            preds = logits.argmax(-1)

            for p_seq, t_seq in zip(preds, y):
                for p, t in zip(p_seq, t_seq):
                    if t != -100:
                        all_p.append(p.item())
                        all_t.append(t.item())

    f1 = f1_score(all_t, all_p, average="macro", zero_division=0)
    p = precision_score(all_t, all_p, average="macro", zero_division=0)
    r = recall_score(all_t, all_p, average="macro", zero_division=0)

    return f1, p, r


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

        f1, p, r = evaluate(model, dev_loader)
        print(f"Dev — F1: {f1:.4f} | P: {p:.4f} | R: {r:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model (F1={best_f1:.4f})")

    # ===== TEST =====
    print("\n=== TEST ===")
    model.load_state_dict(torch.load(save_path, map_location=device))
    f1, p, r = evaluate(model, test_loader)
    print(f"Test — F1: {f1:.4f} | P: {p:.4f} | R: {r:.4f}")


if __name__ == "__main__":
    main()
