import os
import random
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Returns the available device (CUDA or CPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------------------
# Data Processing & Caching
# ------------------------------------------------------------------------------


def feature_engineering(df):
    """
    Applies Augmented Physics-Informed Engineering.
    """
    # Create a copy to avoid SettingWithCopy warnings
    df = df.copy()

    # 1. Cyclical Augmentation (Keep raw Aspect)
    # Aspect is in degrees (0-360)
    df["Aspect_Sin"] = np.sin(df["Aspect"] * np.pi / 180.0)
    df["Aspect_Cos"] = np.cos(df["Aspect"] * np.pi / 180.0)

    # 2. Geometric Magnitude (Euclidean Distance to Hydrology)
    # H_Dist^2 + V_Dist^2
    df["Hydro_Euclidean"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"] ** 2
        + df["Vertical_Distance_To_Hydrology"] ** 2
    )

    # 3. Directional Preservation (Absolute Hydrology Elevation)
    # Elevation - Vertical_Dist
    df["Hydro_Elevation"] = df["Elevation"] - df["Vertical_Distance_To_Hydrology"]

    # 4. Global Context (Mean Distance to Amenities)
    # (Hydro + Road + Fire) / 3
    df["Mean_Amenities_Dist"] = (
        df["Horizontal_Distance_To_Hydrology"]
        + df["Horizontal_Distance_To_Roadways"]
        + df["Horizontal_Distance_To_Fire_Points"]
    ) / 3.0

    return df


def get_data(
    load_cached_data=True,
    batch_size=4096,
    data_dir="./metadata",
    cache_dir="./working/idea_29",
):
    """
    Loads data, performs feature engineering, handles caching, and returns DataLoaders.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Cache file paths
    cache_files = {
        "train_X": os.path.join(cache_dir, "train_X.npy"),
        "train_y": os.path.join(cache_dir, "train_y.npy"),
        "val_X": os.path.join(cache_dir, "val_X.npy"),
        "val_y": os.path.join(cache_dir, "val_y.npy"),
        "test_X": os.path.join(cache_dir, "test_X.npy"),
        "test_ids": os.path.join(cache_dir, "test_ids.npy"),
    }

    # Check if cache exists
    cache_exists = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and cache_exists:
        print("Loading cached data...")
        X_train = np.load(cache_files["train_X"])
        y_train = np.load(cache_files["train_y"])
        X_val = np.load(cache_files["val_X"])
        y_val = np.load(cache_files["val_y"])
        X_test = np.load(cache_files["test_X"])
        test_ids = np.load(cache_files["test_ids"])
    else:
        print("Processing data from scratch...")
        # Load Parquet
        train_df = pd.read_parquet(os.path.join(data_dir, "train.parquet"))
        val_df = pd.read_parquet(os.path.join(data_dir, "val.parquet"))
        test_df = pd.read_parquet(os.path.join(data_dir, "test.parquet"))

        # Extract IDs and Targets
        # Targets are 1-7, convert to 0-6 for PyTorch
        y_train = train_df["Cover_Type"].values - 1
        y_val = val_df["Cover_Type"].values - 1
        test_ids = test_df["Id"].values.astype(np.int64)

        # Drop Id and Target from features
        train_df = train_df.drop(columns=["Id", "Cover_Type"], errors="ignore")
        val_df = val_df.drop(columns=["Id", "Cover_Type"], errors="ignore")
        test_df = test_df.drop(columns=["Id"], errors="ignore")

        # Feature Engineering
        train_df = feature_engineering(train_df)
        val_df = feature_engineering(val_df)
        test_df = feature_engineering(test_df)

        # Identify Columns
        # Binary features: Soil_Type*, Wilderness_Area*
        bin_cols = [
            c for c in train_df.columns if "Soil_Type" in c or "Wilderness_Area" in c
        ]
        cont_cols = [c for c in train_df.columns if c not in bin_cols]

        # Standardization (Fit on Train, Transform all)
        scaler = StandardScaler()
        train_df[cont_cols] = scaler.fit_transform(
            train_df[cont_cols].astype(np.float32)
        )
        val_df[cont_cols] = scaler.transform(val_df[cont_cols].astype(np.float32))
        test_df[cont_cols] = scaler.transform(test_df[cont_cols].astype(np.float32))

        # Convert to Numpy (Float32)
        # Ensure column order is identical
        cols = cont_cols + bin_cols
        X_train = train_df[cols].values.astype(np.float32)
        X_val = val_df[cols].values.astype(np.float32)
        X_test = test_df[cols].values.astype(np.float32)

        # Save to Cache
        np.save(cache_files["train_X"], X_train)
        np.save(cache_files["train_y"], y_train)
        np.save(cache_files["val_X"], X_val)
        np.save(cache_files["val_y"], y_val)
        np.save(cache_files["test_X"], X_test)
        np.save(cache_files["test_ids"], test_ids)

    # Create DataLoaders
    train_dataset = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train).long()
    )
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).long())
    test_dataset = TensorDataset(torch.from_numpy(X_test))  # No targets for test

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, test_ids, X_train.shape[1]


# ------------------------------------------------------------------------------
# Model Architecture
# ------------------------------------------------------------------------------


class VectorCrossLayer(nn.Module):
    """
    Vector-based (Rank-1) Cross Layer with Dot-Product Mixing.
    x_{l+1} = x_0 * (x_l . w) + b + x_l
    """

    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.w = nn.Parameter(torch.empty(input_dim))
        self.b = nn.Parameter(torch.zeros(input_dim))

        # Initialization: Near-Zero to start as identity mapping
        nn.init.normal_(self.w, mean=0, std=1e-4)

    def forward(self, x0, xl):
        # x0: [batch, dim]
        # xl: [batch, dim]
        # w: [dim]

        # Dot product (batch-wise): (xl * w).sum(dim=1) -> [batch]
        # We broadcast this scalar to [batch, dim]
        dot_prod = (xl * self.w).sum(dim=1, keepdim=True)  # [batch, 1]

        # x0 * scalar
        interaction = x0 * dot_prod  # [batch, dim]

        return interaction + self.b + xl


class PreActResBlock(nn.Module):
    """
    Full Pre-Activation ResNet Block.
    BN -> ReLU -> Dropout -> Linear -> BN -> ReLU -> Dropout -> Linear -> Add
    """

    def __init__(self, dim, dropout_rate=0.2):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lin1 = nn.Linear(dim, dim)

        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.lin2 = nn.Linear(dim, dim)

    def forward(self, x):
        out = self.bn1(x)
        out = F.relu(out)
        out = self.dropout1(out)
        out = self.lin1(out)

        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.lin2(out)

        return x + out


class DeepParallelVectorDCNResNet(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes=7,
        hidden_dim=512,
        num_cross_layers=3,
        num_res_blocks=4,
        dropout_rate=0.2,
    ):
        super().__init__()

        # Branch 1: Vector DCN
        self.cross_layers = nn.ModuleList(
            [VectorCrossLayer(input_dim) for _ in range(num_cross_layers)]
        )

        # Branch 2: Deep ResNet
        # Projection to hidden dim
        self.res_proj = nn.Linear(input_dim, hidden_dim)
        self.res_blocks = nn.ModuleList(
            [PreActResBlock(hidden_dim, dropout_rate) for _ in range(num_res_blocks)]
        )

        # Combination Head
        # Concat: input_dim (from DCN) + hidden_dim (from ResNet)
        concat_dim = input_dim + hidden_dim
        self.head = nn.Linear(concat_dim, num_classes)

    def forward(self, x):
        # x: [batch, input_dim]

        # Branch 1: DCN
        x_dcn = x
        for layer in self.cross_layers:
            x_dcn = layer(x, x_dcn)  # Pass x0 (x) and xl (x_dcn)

        # Branch 2: ResNet
        x_res = self.res_proj(x)
        for block in self.res_blocks:
            x_res = block(x_res)

        # Combine
        x_concat = torch.cat([x_dcn, x_res], dim=1)
        logits = self.head(x_concat)

        return logits


# ------------------------------------------------------------------------------
# Training & Evaluation
# ------------------------------------------------------------------------------


def train_model(
    model, train_loader, val_loader, device, epochs=60, lr=1e-3, patience=10
):
    criterion = nn.CrossEntropyLoss()
    # Decoupled Weight Decay (AdamW)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    # ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    print(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss /= total
        train_acc = correct / total

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.6f}"
        )

        # Scheduler step
        scheduler.step(val_acc)

        # Early Stopping & Checkpointing
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Load best weights
    model.load_state_dict(best_model_wts)
    print(f"Best Validation Accuracy: {best_acc:.6f}")
    return model


def generate_submission(
    model, test_loader, test_ids, device, output_path="./submission/submission.csv"
):
    model.eval()
    predictions = []

    print("Generating predictions...")
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs[0].to(device)  # TensorDataset returns tuple (data,)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            # Convert back to 1-based indexing (0-6 -> 1-7)
            predicted = predicted + 1
            predictions.extend(predicted.cpu().numpy())

    # Create submission DF
    sub_df = pd.DataFrame({"Id": test_ids, "Cover_Type": predictions})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sub_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


# ------------------------------------------------------------------------------
# Main Pipeline Function
# ------------------------------------------------------------------------------


def run_pipeline(epochs=60, batch_size=4096):
    seed_everything(42)
    device = get_device()

    # Data
    train_loader, val_loader, test_loader, test_ids, input_dim = get_data(
        batch_size=batch_size
    )

    # Model
    model = DeepParallelVectorDCNResNet(input_dim=input_dim).to(device)

    # Train
    model = train_model(model, train_loader, val_loader, device, epochs=epochs)

    # Submit
    generate_submission(model, test_loader, test_ids, device)
