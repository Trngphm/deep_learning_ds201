from torch.utils.data import DataLoader
from dataset import VinaFood, collate_fn

train_dataset = VinaFood(
    path="./VinaFood21/train",
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)

# Kiểm tra
item = next(iter(train_loader))
print(item["image"].shape)  # (32,3,224,224)
print(item["label"].shape)  # (32,)
