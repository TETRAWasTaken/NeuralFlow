import os
import csv
import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. Model Definition (1:1 with Rust DeepNet)
# ==========================================
class DeepNet(nn.Module):
    def __init__(self, in_features: int = 9, hidden_dim: int = 64, out_features: int = 1):
        super().__init__()
        # In NeuralFlow, Linear has no bias and weights are initialized uniformly in [-1, 1]
        self.fc1 = nn.Linear(in_features, hidden_dim, bias=False)
        self.ln1 = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.drop1 = nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ln2 = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.drop2 = nn.Dropout(p=0.2)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ln3 = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.drop3 = nn.Dropout(p=0.2)

        self.fc4 = nn.Linear(hidden_dim, out_features, bias=False)
        self.relu = nn.ReLU()

        self._init_weights()

    def _init_weights(self):
        # Match Tensor::random in NeuralFlow: uniform(-1.0, 1.0)
        for m in [self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.uniform_(m.weight, -1.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.drop1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.ln2(x)
        x = self.drop2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.ln3(x)
        x = self.drop3(x)
        x = self.relu(x)

        x = self.fc4(x)
        return x


# ==========================================
# 2. Data Loading & Standard Scaler
# ==========================================
class StandardScaler:
    def __init__(self, eps: float = 1e-8):
        self.mean = None
        self.std = None
        self.eps = eps
        self.is_fitted = False

    def fit(self, data: np.ndarray):
        # Population mean and std (matching NeuralFlow's variance formula)
        self.mean = np.mean(data, axis=0)
        variance = np.mean(data ** 2, axis=0) - (self.mean ** 2)
        variance = np.maximum(variance, 0.0)
        self.std = np.sqrt(variance) + self.eps
        self.is_fitted = True

    def transform(self, data: np.ndarray) -> np.ndarray:
        assert self.is_fitted, "Scaler has not been fitted yet"
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        assert self.is_fitted, "Scaler has not been fitted yet"
        return (data * self.std) + self.mean


class HousingDataset(Dataset):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def load_housing_csv(csv_path: str, target_col_idx: int = 8, has_header: bool = True):
    inputs = []
    targets = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        if has_header:
            next(reader, None)

        for row in reader:
            if not row:
                continue

            # In Rust CsvDataset, invalid floats (e.g. 'NEAR BAY') default to 0.0
            row_values = []
            for val in row:
                try:
                    row_values.append(float(val.strip()))
                except ValueError:
                    row_values.append(0.0)

            target = [row_values[target_col_idx]]
            input_feat = [v for idx, v in enumerate(row_values) if idx != target_col_idx]

            inputs.append(input_feat)
            targets.append(target)

    return np.array(inputs, dtype=np.float32), np.array(targets, dtype=np.float32)


# ==========================================
# 3. Metrics Evaluator (Matching Rust Evaluator)
# ==========================================
def calculate_metrics(y_pred: np.ndarray, y_true: np.ndarray):
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y_true))

    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-8)) if ss_tot > 0 else 0.0

    return {"loss": mse, "r2": r2, "mae": mae, "rmse": rmse}


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module):
    model.eval()
    total_loss = 0.0
    batch_count = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            preds = model(x_batch)
            loss = criterion(preds, y_batch)

            total_loss += loss.item()
            batch_count += 1

            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    metrics = calculate_metrics(all_preds, all_targets)
    metrics["loss"] = total_loss / batch_count if batch_count > 0 else 0.0
    return metrics


# ==========================================
# 4. Main Training Pipeline
# ==========================================
def main():
    in_features = 9
    hidden_dim = 64
    out_features = 1
    learning_rate = 5e-2
    batch_size = 256
    epochs = 50
    checkpoint_path = "best_deepnet_py.pt"

    # Set seed for reproducible comparison
    np.random.seed(42)
    torch.manual_seed(42)

    print("--- Loading Dataset ---")
    csv_path = os.path.join(os.path.dirname(__file__), "housing.csv")
    inputs, targets = load_housing_csv(csv_path, target_col_idx=8, has_header=True)
    total_samples = len(inputs)
    print(f"Loaded {total_samples} total samples.")

    # Standard Scaling
    x_scaler = StandardScaler()
    x_scaler.fit(inputs)
    inputs_scaled = x_scaler.transform(inputs)

    y_scaler = StandardScaler()
    y_scaler.fit(targets)
    targets_scaled = y_scaler.transform(targets)

    # Train / Val / Test split (70% / 15% / 15%)
    indices = np.random.permutation(total_samples)
    train_end = int(0.70 * total_samples)
    val_end = int(0.85 * total_samples)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    print(
        f"Dataset split: {len(train_idx)} train, {len(val_idx)} validation, {len(test_idx)} test samples."
    )

    train_dataset = HousingDataset(inputs_scaled[train_idx], targets_scaled[train_idx])
    val_dataset = HousingDataset(inputs_scaled[val_idx], targets_scaled[val_idx])
    test_dataset = HousingDataset(inputs_scaled[test_idx], targets_scaled[test_idx])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = DeepNet(in_features, hidden_dim, out_features)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")

    print("\n--- Starting Training ---")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        train_batches = 0

        for inputs_batch, targets_batch in train_loader:
            optimizer.zero_grad()
            pred = model(inputs_batch)
            loss = criterion(pred, targets_batch)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            if not math.isnan(loss_val):
                total_train_loss += loss_val
                train_batches += 1

        avg_train_loss = total_train_loss / train_batches if train_batches > 0 else 0.0
        val_metrics = evaluate(model, val_loader, criterion)
        val_loss = val_metrics["loss"]

        # Checkpoint best model
        if val_loss < best_val_loss:
            print(f"Loss improved from {best_val_loss:.5f} to {val_loss:.5f}. Saving checkpoint to '{checkpoint_path}'")
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "metrics": val_metrics,
                },
                checkpoint_path,
            )

        print(
            f"Epoch {epoch:2d}/{epochs} - Train MSE: {avg_train_loss:.4e} | "
            f"Val: Loss: {val_loss:.5f} | R2: {val_metrics['r2']:.4f} | "
            f"MAE: {val_metrics['mae']:.4f} | RMSE: {val_metrics['rmse']:.4f}"
        )

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f}s")

    # ==========================================
    # 5. Testing with Best Checkpoint
    # ==========================================
    print("\n--- Loading Best Checkpoint for Testing ---")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model from epoch {checkpoint['epoch']} with best val loss: {checkpoint['val_loss']:.4e}")

    test_metrics = evaluate(model, test_loader, criterion)
    print(
        f"Test Set Evaluation -> Loss: {test_metrics['loss']:.5f} | R2: {test_metrics['r2']:.4f} | "
        f"MAE: {test_metrics['mae']:.4f} | RMSE: {test_metrics['rmse']:.4f}"
    )

    # ==========================================
    # 6. Sample Inference
    # ==========================================
    print("\n--- Inference on Sample Test Data ---")
    model.eval()
    sample_count = min(5, len(test_dataset))
    with torch.no_grad():
        for i in range(sample_count):
            sample_inp, sample_tgt = test_dataset[i]
            pred_scaled = model(sample_inp.unsqueeze(0)).item()
            actual_scaled = sample_tgt.item()

            pred_val = y_scaler.inverse_transform(np.array([[pred_scaled]]))[0, 0]
            actual_val = y_scaler.inverse_transform(np.array([[actual_scaled]]))[0, 0]
            abs_err = abs(pred_val - actual_val)

            print(
                f"Sample #{i + 1}: Predicted = ${pred_val:,.2f} | "
                f"Actual = ${actual_val:,.2f} | Absolute Error = ${abs_err:,.2f}"
            )


if __name__ == "__main__":
    main()
