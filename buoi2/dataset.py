import torch
from torch.utils.data import Dataset
import cv2 as cv
import os
from torchvision import transforms

def collate_fn(samples):
    images = [sample["image"].unsqueeze(0) for sample in samples]
    labels = [sample["label"] for sample in samples]

    images = torch.cat(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)

    return {
        "image": images,
        "label": labels
    }

class VinaFood(Dataset):
    def __init__(self, path, image_size=(224, 224)):
        super().__init__()

        self.image_size = image_size
        self.label2idx = {}
        self.idx2label = {}
        self.data = self.load_data(path)

        self.transform = transforms.Compose([
            transforms.ToTensor(),  # chuyển (H, W, C) -> (C, H, W) và scale [0,255] -> [0,1]
        ])

    def load_data(self, path):
        data = []
        label_id = 0

        for folder in sorted(os.listdir(path)):
            folder_path = os.path.join(path, folder)
            if not os.path.isdir(folder_path):
                continue

            label = folder
            if label not in self.label2idx:
                self.label2idx[label] = label_id
                label_id += 1

            for image_file in os.listdir(folder_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(folder_path, image_file)
                    data.append({
                        "path": image_path,
                        "label": label
                    })

        self.idx2label = {v: k for k, v in self.label2idx.items()}
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        image = cv.imread(item["path"])
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.resize(image, self.image_size)

        image = self.transform(image)
        label_id = self.label2idx[item["label"]]

        return {
            "image": image,
            "label": label_id
        }
