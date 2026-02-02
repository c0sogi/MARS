import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import copy
import random
import gc
from torchvision.ops import stochastic_depth

# ------------------------------------------------------------------------------
# 1. Utility Functions
# ------------------------------------------------------------------------------


def seed_everything(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Returns the appropriate device (CUDA or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ------------------------------------------------------------------------------
# 2. Data Processing & Dataset
# ------------------------------------------------------------------------------


class ForestCoverDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def process_data(load_cached_data=True, cache_dir="./working/idea_11/"):
    """
    Loads data from metadata, performs feature engineering, and prepares tensors.
    Implements caching to avoid re-processing.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Cache file paths
    cache_train_X = os.path.join(cache_dir, "train_X.npy")
    cache_train_y = os.path.join(cache_dir, "train_y.npy")
    cache_val_X = os.path.join(cache_dir, "val_X.npy")
    cache_val_y = os.path.join(cache_dir, "val_y.npy")
    cache_test_X = os.path.join(cache_dir, "test_X.npy")
    cache_test_ids = os.path.join(cache_dir, "test_ids.npy")

    # Check cache
    if load_cached_data:
        if (
            os.path.exists(cache_train_X)
            and os.path.exists(cache_train_y)
            and os.path.exists(cache_val_X)
            and os.path.exists(cache_val_y)
            and os.path.exists(cache_test_X)
            and os.path.exists(cache_test_ids)
        ):
            print("Loading cached data...")
            train_X = np.load(cache_train_X)
            train_y = np.load(cache_train_y)
            val_X = np.load(cache_val_X)
            val_y = np.load(cache_val_y)
            test_X = np.load(cache_test_X)
            test_ids = np.load(cache_test_ids)
            return train_X, train_y, val_X, val_y, test_X, test_ids

    print("Processing data from scratch...")

    # Load Metadata Parquet
    df_train = pd.read_parquet("./metadata/train.parquet")
    df_val = pd.read_parquet("./metadata/val.parquet")
    df_test = pd.read_parquet("./metadata/test.parquet")

    # Feature Engineering Function
    def engineer_features(df):
        # 1. Augmentation: Aspect Sin/Cos
        df["Aspect_Sin"] = np.sin(np.radians(df["Aspect"]))
        df["Aspect_Cos"] = np.cos(np.radians(df["Aspect"]))

        # 2. Geometric Magnitude: Euclidean Dist to Hydrology
        df["Hydrology_Dist"] = np.sqrt(
            df["Horizontal_Distance_To_Hydrology"] ** 2
            + df["Vertical_Distance_To_Hydrology"] ** 2
        )

        # 3. Directional Preservation: Absolute Hydrology Elevation
        df["Abs_Hydro_Elev"] = df["Elevation"] - df["Vertical_Distance_To_Hydrology"]

        # 4. Global Context: Mean Distance to Amenities
        df["Mean_Dist_Amenities"] = df[
            [
                "Horizontal_Distance_To_Hydrology",
                "Horizontal_Distance_To_Roadways",
                "Horizontal_Distance_To_Fire_Points",
            ]
        ].mean(axis=1)

        return df

    # Apply Engineering
    df_train = engineer_features(df_train)
    df_val = engineer_features(df_val)
    df_test = engineer_features(df_test)

    # Prepare Columns
    target_col = "Cover_Type"
    id_col = "Id"
    drop_cols = [target_col, id_col]

    feature_cols = [c for c in df_train.columns if c not in drop_cols]

    # Identify Binary vs Continuous
    # Binary columns contain 'Soil_Type' or 'Wilderness_Area'
    binary_cols = [
        c for c in feature_cols if "Soil_Type" in c or "Wilderness_Area" in c
    ]
    continuous_cols = [c for c in feature_cols if c not in binary_cols]

    # Standardization (Fit on Train, Transform All)
    scaler = StandardScaler()
    scaler.fit(df_train[continuous_cols])

    def transform_df(df, is_test=False):
        # Continuous
        cont_data = scaler.transform(df[continuous_cols])
        # Binary (Raw)
        bin_data = df[binary_cols].values
        # Concatenate
        X = np.hstack([cont_data, bin_data])

        if is_test:
            return X, df[id_col].values
        else:
            # Adjust targets to 0-indexed (Class 1-7 -> 0-6)
            y = df[target_col].values - 1
            return X, y

    train_X, train_y = transform_df(df_train)
    val_X, val_y = transform_df(df_val)
    test_X, test_ids = transform_df(df_test, is_test=True)

    # Save to cache
    np.save(cache_train_X, train_X)
    np.save(cache_train_y, train_y)
    np.save(cache_val_X, val_X)
    np.save(cache_val_y, val_y)
    np.save(cache_test_X, test_X)
    np.save(cache_test_ids, test_ids)

    return train_X, train_y, val_X, val_y, test_X, test_ids


# ------------------------------------------------------------------------------
# 3. Model Architecture
# ------------------------------------------------------------------------------


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (Batch, Channels)
        # Squeeze is Identity for 1D vectors
        y = self.fc1(x)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y


class ResBlock(nn.Module):
    def __init__(self, in_features, hidden_features, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(in_features)
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.norm2 = nn.BatchNorm1d(hidden_features)
        self.linear2 = nn.Linear(hidden_features, in_features)

        self.se = SEBlock(in_features)
        self.drop_path_prob = drop_path

    def forward(self, x):
        identity = x
        out = self.norm1(x)
        out = self.linear1(out)
        out = self.act(out)
        out = self.dropout(out)

        out = self.norm2(out)
        out = self.linear2(out)

        out = self.se(out)

        if self.training and self.drop_path_prob > 0.0:
            out = stochastic_depth(
                out, self.drop_path_prob, mode="batch", training=True
            )

        return identity + out


class DCNv2Vector(nn.Module):
    """
    Vector-based Deep & Cross Network Layer.
    Implements Dot-Product Mixing: x_{l+1} = x_0 * (x_l^T w) + b + x_l
    """

    def __init__(self, input_dim, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim

        self.W = nn.ParameterList(
            [nn.Parameter(torch.randn(input_dim)) for _ in range(num_layers)]
        )
        self.b = nn.ParameterList(
            [nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)]
        )

        for w in self.W:
            nn.init.xavier_normal_(w.unsqueeze(0))

    def forward(self, x):
        x0 = x
        xl = x
        for i in range(self.num_layers):
            # dot: (Batch, D) * (D,) -> (Batch, D) -> sum -> (Batch, 1)
            dot = (xl * self.W[i]).sum(dim=1, keepdim=True)
            xl = x0 * dot + self.b[i] + xl
        return xl


class ParallelDCN_SE_ResNet(nn.Module):
    def __init__(self, input_dim, num_classes=7):
        super().__init__()

        # Branch 1: DCN
        self.dcn = DCNv2Vector(input_dim, num_layers=3)

        # Branch 2: SE-ResNet
        self.stem = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU()
        )

        self.blocks = nn.ModuleList(
            [ResBlock(512, 512, drop_path=0.1) for _ in range(3)]
        )

        # Combination
        self.head = nn.Sequential(
            nn.BatchNorm1d(input_dim + 512), nn.Linear(input_dim + 512, num_classes)
        )

    def forward(self, x):
        # Branch 1
        x_dcn = self.dcn(x)

        # Branch 2
        x_res = self.stem(x)
        for block in self.blocks:
            x_res = block(x_res)

        # Combine
        x_cat = torch.cat([x_dcn, x_res], dim=1)
        out = self.head(x_cat)
        return out


# ------------------------------------------------------------------------------
# 4. Training & Orchestration
# ------------------------------------------------------------------------------


def run_idea_10(epochs=60, batch_size=4096, quick_run=False):
    """
    Main function to execute the pipeline: Data Loading, Training, Inference.
    """
    # Setup
    seed_everything(42)
    device = get_device()
    print(f"Using device: {device}")

    # Data
    train_X, train_y, val_X, val_y, test_X, test_ids = process_data()

    if quick_run:
        train_X = train_X[:10000]
        train_y = train_y[:10000]
        val_X = val_X[:2000]
        val_y = val_y[:2000]
        epochs = 2

    # Datasets
    train_ds = ForestCoverDataset(train_X, train_y)
    val_ds = ForestCoverDataset(val_X, val_y)
    test_ds = ForestCoverDataset(test_X)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # Model
    input_dim = train_X.shape[1]
    num_classes = 7
    model = ParallelDCN_SE_ResNet(input_dim, num_classes).to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience = 8
    patience_counter = 0

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
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

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        # Print full precision as requested
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} Acc: {train_acc} | Val Loss: {val_loss} Acc: {val_acc}"
        )

        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Inference
    print("Generating predictions...")
    model.load_state_dict(best_model_wts)
    model.eval()

    preds = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            preds.extend(predicted.cpu().numpy())

    # Remap 0-6 back to 1-7
    final_preds = np.array(preds) + 1

    # Save Submission
    os.makedirs("./submission", exist_ok=True)
    sub_df = pd.DataFrame({"Id": test_ids, "Cover_Type": final_preds})
    sub_df.to_csv("./submission/submission.csv", index=False)
    print(f"Submission saved to ./submission/submission.csv with {len(sub_df)} rows.")
