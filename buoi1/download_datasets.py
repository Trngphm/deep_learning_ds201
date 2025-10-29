from torchvision import datasets, transforms

# Chuyển ảnh thành Tensor (bắt buộc cho PyTorch)
transform = transforms.ToTensor()

# Tải tập huấn luyện
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

# Tải tập kiểm tra
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    transform=transform,
    download=True
)

print("Số ảnh train:", len(train_dataset))
print("Số ảnh test:", len(test_dataset))
