import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm
import argparse

from dataset import collate_fn, VinaFood
from googLeNet import GoogLeNet  
from resnet import ResNet18 
from pretrain_resnet import PretrainedResnet


device = "cuda" if torch.cuda.is_available() else "cpu"


# ==================== Hàm đánh giá ====================
def evaluate(dataloader, model, idx2label):
    model.eval()
    predictions = []
    trues = []

    with torch.no_grad():
        for item in dataloader:
            image = item["image"].to(device).float()
            label = item["label"].to(device)
            output = model(image)
            output = output.argmax(dim=-1)

            predictions.extend(output.cpu().tolist())
            trues.extend(label.cpu().tolist())

    class_report = classification_report(
        trues, predictions,
        target_names=[idx2label[i] for i in sorted(idx2label.keys())],
        output_dict=True
    )

    return {
        "precision": precision_score(trues, predictions, average="macro"),
        "recall": recall_score(trues, predictions, average="macro"),
        "f1": f1_score(trues, predictions, average="macro"),
        "report": class_report
    }


# ==================== Hàm chính ====================
def main(model_name):
    # ---- 1. Load dataset ----
    train_dataset = VinaFood(path="VinaFood21/train")
    test_dataset = VinaFood(path="VinaFood21/test")

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    num_classes = len(train_dataset.label2idx)
    print(f"Số lớp trong dataset: {num_classes}")

    # ---- 2. Khởi tạo mô hình ----
    if model_name == "googlenet":
        model = GoogLeNet(num_classes=num_classes).to(device)
        model_filename = "googlenet_vinafood.pth"
    elif model_name == "resnet":
        model = ResNet18(num_classes=num_classes).to(device)
        model_filename = "resnet18_vinafood.pth"
    elif model_name == "pretrain_resnet":
        model = PretrainedResnet().to(device)
        model_filename = "pretrained_resnet_vinafood.pth"
    else:
        raise ValueError("❌ Model không hợp lệ. Hãy chọn 'googlenet' hoặc 'resnet'.")

    # ---- 3. Loss + Optimizer ----
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ---- 4. Train loop ----
    EPOCHS = 5
    for epoch in range(EPOCHS):
        print(f"\n===== Epoch {epoch + 1}/{EPOCHS} =====")
        model.train()
        losses = []

        progress_bar = tqdm(train_dataloader, desc=f"Training epoch {epoch + 1}", leave=False)
        for batch in progress_bar:
            image = batch["image"].to(device).float()
            label = batch["label"].to(device)

            output = model(image)
            loss = loss_fn(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            progress_bar.set_postfix({"loss": f"{np.mean(losses):.4f}"})

        print(f"Loss trung bình: {np.mean(losses):.4f}")

        # ---- 5. Evaluate ----
        scores = evaluate(test_dataloader, model, train_dataset.idx2label)

        print(f"Precision (macro): {scores['precision']:.4f}")
        print(f"Recall (macro):    {scores['recall']:.4f}")
        print(f"F1 (macro):        {scores['f1']:.4f}")

        print("\nĐánh giá theo từng lớp:")
        for cls_name, metrics in scores["report"].items():
            if isinstance(metrics, dict):
                print(f"  {cls_name:15s} | "
                      f"P={metrics['precision']:.3f}, "
                      f"R={metrics['recall']:.3f}, "
                      f"F1={metrics['f1-score']:.3f}")

    # ---- 6. Lưu model ----
    torch.save(model.state_dict(), model_filename)
    print(f"\n✅ Đã lưu model thành công vào '{model_filename}'")


# ==================== Entry point ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình VinaFood21")
    parser.add_argument("model", choices=["googlenet", "resnet", "pretrain_resnet"], help="Chọn mô hình huấn luyện")
    args = parser.parse_args()

    main(args.model)
