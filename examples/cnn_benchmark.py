import os
import struct
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# 1. MNIST Binary Loader (Matches load_mnist in Rust 1:1)
# ==============================================================================
class BinaryMNISTDataset(Dataset):
    def __init__(self, image_path: str, label_path: str, max_samples: int = None):
        with open(image_path, "rb") as f_img, open(label_path, "rb") as f_lbl:
            # Read Image Header
            magic_img, total_imgs, rows, cols = struct.unpack(">IIII", f_img.read(16))
            magic_lbl, total_lbls = struct.unpack(">II", f_lbl.read(8))
            assert total_imgs == total_lbls, "Image and label count mismatch"

            count = min(max_samples, total_imgs) if max_samples else total_imgs
            image_pixels = rows * cols  # 784

            # Read and normalize pixels to [0.0, 1.0] in float32
            raw_images = np.frombuffer(f_img.read(count * image_pixels), dtype=np.uint8)
            self.images = raw_images.reshape(count, 1, rows, cols).astype(np.float32) / 255.0

            raw_labels = np.frombuffer(f_lbl.read(count), dtype=np.uint8)
            self.labels = raw_labels.astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.images[idx]), torch.tensor(self.labels[idx])


# ==============================================================================
# 2. Benchmark CNN Architecture (1:1 with NeuralFlow build_cnn_model)
# ==============================================================================
class BenchmarkCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Conv Block 1: (N, 1, 28, 28) -> (N, 16, 14, 14)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 2: (N, 16, 14, 14) -> (N, 32, 7, 7)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten & Classification Head
        self.flatten = nn.Flatten()
        # In NeuralFlow, Linear has no bias
        self.fc1 = nn.Linear(32 * 7 * 7, 128, bias=False)
        self.relu3 = nn.ReLU()
        self.drop = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(128, num_classes, bias=False)

        self._init_weights()

    def _init_weights(self):
        # Match NeuralFlow's Kaiming Normal (fan_in, nonlinearity='relu')
        for conv in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(conv.weight, mode='fan_in', nonlinearity='relu')
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

        for fc in [self.fc1, self.fc2]:
            nn.init.kaiming_normal_(fc.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.drop(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==============================================================================
# 3. Main Benchmark Runner
# ==============================================================================
def main():
    print("============================================================")
    print("  PyTorch: MNIST CNN Training & Benchmark")
    print("============================================================")

    device = torch.device("cpu")
    print(f"Device: {device}")

    # Paths to MNIST files
    train_images_path = "data/mnist/train-images-idx3-ubyte"
    train_labels_path = "data/mnist/train-labels-idx1-ubyte"
    test_images_path = "data/mnist/t10k-images-idx3-ubyte"
    test_labels_path = "data/mnist/t10k-labels-idx1-ubyte"

    # Dataset size configuration (Matches NeuralFlow 1:1)
    train_subset_size = 2000
    test_subset_size = 1000

    print("\nLoading MNIST dataset...")
    load_start = time.perf_counter()
    train_dataset = BinaryMNISTDataset(train_images_path, train_labels_path, train_subset_size)
    test_dataset = BinaryMNISTDataset(test_images_path, test_labels_path, test_subset_size)
    print(f"Loaded {len(train_dataset)} train samples, {len(test_dataset)} test samples in {time.perf_counter() - load_start:.2f}s")

    # Hyperparameters (matched 1:1 with NeuralFlow)
    batch_size = 64
    epochs = 20
    learning_rate = 0.001

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Build Model
    model = BenchmarkCNN(num_classes=10).to(device)
    total_params = count_parameters(model)
    print(f"Model initialized with {total_params:,} trainable parameters")
    print(f"Optimizer: Adam (lr={learning_rate}) | Loss: CrossEntropyLoss")
    print(f"Batch size: {batch_size} | Epochs: {epochs}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    print("-" * 75)
    print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Train Acc':<9} | {'Val Loss':<10} | {'Val Acc':<8} | {'Throughput':<12}")
    print("-" * 75)

    overall_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        # Training Phase
        model.train()
        total_train_loss = 0.0
        train_correct = 0
        train_samples = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            current_b = x_batch.size(0)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * current_b
            preds = logits.argmax(dim=1)
            train_correct += (preds == y_batch).sum().item()
            train_samples += current_b

        epoch_duration = time.perf_counter() - epoch_start
        samples_per_sec = train_samples / epoch_duration
        avg_train_loss = total_train_loss / train_samples
        train_accuracy = (train_correct / train_samples) * 100.0

        # Validation Phase
        model.eval()
        total_val_loss = 0.0
        val_correct = 0
        val_samples = 0

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                current_b = x_batch.size(0)

                logits = model(x_batch)
                loss = criterion(logits, y_batch)

                total_val_loss += loss.item() * current_b
                preds = logits.argmax(dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_samples += current_b

        avg_val_loss = total_val_loss / val_samples
        val_accuracy = (val_correct / val_samples) * 100.0

        print(
            f"{f'{epoch}/{epochs}':<6} | {avg_train_loss:<10.4f} | {train_accuracy:<8.2f}% | "
            f"{avg_val_loss:<10.4f} | {val_accuracy:<7.2f}% | {samples_per_sec:<7.1f} img/s"
        )

    print("-" * 75)
    print(f"\nBenchmark completed in {time.perf_counter() - overall_start:.2f}s")


if __name__ == "__main__":
    main()
