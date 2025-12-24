import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import argparse
import numpy as np
import os

# =====================
# IMPORT PROJECT FILES
# =====================
from vocab.uit_viocd import Vocab, UITViOCDDataset, collate_fn
from models.classification import TransformerClassifier
from config.uit_viocd import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, SYSTEM_CONFIG

# =====================
# DEVICE & SEED
# =====================
device = SYSTEM_CONFIG["device"]
if device == "cuda" and not torch.cuda.is_available():
    device = "cpu"

torch.manual_seed(SYSTEM_CONFIG["seed"])

# =====================
# EVALUATION FUNCTION
# =====================
def evaluate(model, loader, id2label):
    model.eval()
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for batch in loader:
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)

            logits = model(x)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_trues.extend(y.cpu().tolist())

    acc = accuracy_score(all_trues, all_preds)
    f1 = f1_score(all_trues, all_preds, average="macro", zero_division=0)

    report = classification_report(
        all_trues,
        all_preds,
        target_names=[id2label[i] for i in sorted(id2label.keys())],
        zero_division=0
    )

    return acc, f1, report

# =====================
# MAIN
# =====================
def main():
    dataset_name = DATA_CONFIG.get("dataset_name", "default")

    save_dir = os.path.join("checkpoints", dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "transformer.pth")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save",
        type=str,
        default=save_path
    )
    args = parser.parse_args()
    

    # =====================
    # LOAD DATA + VOCAB
    # =====================
    train_ds = UITViOCDDataset(DATA_CONFIG["train_file"])
    vocab = train_ds.vocab

    dev_ds = UITViOCDDataset(
        DATA_CONFIG["dev_file"],
        vocab=vocab
    )
    test_ds = UITViOCDDataset(
        DATA_CONFIG["test_file"],
        vocab=vocab
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=SYSTEM_CONFIG["num_workers"]
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

    print(f"Vocab size: {vocab.vocab_size}")
    print(f"Num domains: {vocab.n_labels}")

    # =====================
    # MODEL
    # =====================
    model = TransformerClassifier(
        vocab_size=vocab.vocab_size,
        num_labels=vocab.n_labels,
        pad_idx=vocab.word2id[vocab.pad],
        d_model=MODEL_CONFIG["d_model"],
        n_head=MODEL_CONFIG["n_head"],
        n_layer=MODEL_CONFIG["n_layer"],
        d_ff=MODEL_CONFIG["d_ff"],
        dropout=MODEL_CONFIG["dropout"]
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=TRAIN_CONFIG["learning_rate"],
        weight_decay=TRAIN_CONFIG["weight_decay"]
    )

    loss_fn = nn.CrossEntropyLoss()

    # =====================
    # TRAINING LOOP
    # =====================
    best_f1 = 0.0

    for epoch in range(TRAIN_CONFIG["epochs"]):
        model.train()
        losses = []

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{TRAIN_CONFIG['epochs']}",
            leave=True
        )

        for batch in pbar:
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)

            logits = model(x)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()

            if TRAIN_CONFIG["grad_clip"] is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    TRAIN_CONFIG["grad_clip"]
                )

            optimizer.step()

            losses.append(loss.item())
            pbar.set_postfix({"loss": f"{np.mean(losses):.4f}"})

        print(f"Epoch {epoch + 1} - Train loss: {np.mean(losses):.4f}")

        # =====================
        # DEV EVAL
        # =====================
        acc, f1, _ = evaluate(
            model,
            dev_loader,
            vocab.id2domain
        )

        print(f"Dev — Acc: {acc:.4f} | Macro-F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), args.save)
            print(f"Saved best model (Macro-F1={best_f1:.4f})")

    # =====================
    # FINAL TEST
    # =====================
    print("\n=== Final Test ===")
    model.load_state_dict(
        torch.load(args.save, map_location=device)
    )

    acc, f1, report = evaluate(
        model,
        test_loader,
        vocab.id2domain
    )

    print(f"Test — Acc: {acc:.4f} | Macro-F1: {f1:.4f}")
    print("\nClassification Report:")
    print(report)


if __name__ == "__main__":
    main()
