import torch
from torch.utils.data import Dataset
import idx2numpy
import numpy as np

def collate_fn(items):
    data = {
        "image": np.stack([item["image"] for item in items], axis = 0),
        "label": np.stack([item["label"] for item in items], axis = 0),
    }

    data = {
        "image": torch.tensor(data["image"]),
        "label": torch.tensor(data["label"])
    }

    return data

class MnistDataset(Dataset):
    """Some Information about MnistDataset"""
    def __init__(self, image_path, label_path):
        images = idx2numpy.convert_from_file(image_path)
        labels = idx2numpy.convert_from_file(label_path)

        self._data = [{
            "image": np.array(image, dtype=int),
            "label": label
        } for image, label in zip(images.tolist(), labels.tolist())]

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)