import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

from dataset import collate_fn, MnistDataset
from lenet import LeNetModel

device = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(dataloader, model):
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

    class_report = classification_report(trues, predictions, output_dict=True)

    return {
        "precision": precision_score(trues, predictions, average="macro"),
        "recall": recall_score(trues, predictions, average="macro"),
        "f1": f1_score(trues, predictions, average="macro"),
        "report": class_report
    }

if __name__ == "__main__":
    train_dataset = MnistDataset(
        image_path="data/train-images-idx3-ubyte",
        label_path='data/train-labels-idx1-ubyte'
    )

    test_dataset = MnistDataset(
        image_path="data/t10k-images-idx3-ubyte",
        label_path="data/t10k-labels-idx1-ubyte"
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    model = LeNetModel().to(device)
    
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    EPOCHS = 10
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}: ")
        losses = []
        model.train()

        for item in train_dataloader:
            image = item["image"].to(device).float()
            label = item["label"].to(device)

            output = model(image)
            loss = loss_fn(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f"Loss: {np.mean(losses):.4f}")

        scores = evaluate(test_dataloader, model)
        for metric in ["precision", "recall", "f1"]:
            print(f"\t- {metric}: {scores[metric]:.4f}")


        # theo từng lớp
        print("\nĐánh giá theo từng lớp (digit 0–9):")
        for digit in range(10):
            cls = str(digit)
            if cls in scores["report"]:
                print(f"  Digit {digit}: "
                    f"Precision={scores['report'][cls]['precision']:.4f}, "
                    f"Recall={scores['report'][cls]['recall']:.4f}, "
                    f"F1={scores['report'][cls]['f1-score']:.4f}")
