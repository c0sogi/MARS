import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import copy


# ==========================================
# Configuration
# ==========================================
class Config:
    # Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_25"
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model Hyperparameters
    HIDDEN_DIM = 512
    DROPOUT = 0.2
    NUM_BLOCKS = 4
    NUM_CLASSES = 7  # Cover_Type 1-7 (mapped to 0-6 internally)

    # Training Settings
    BATCH_SIZE = 4096
    EPOCHS = 60
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    PATIENCE = 5
    FACTOR = 0.5

    # Seeds
    SEED = 42


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


set_seed(Config.SEED)


# ==========================================
# Data Processing & Feature Engineering
# ==========================================
def feature_engineering(df):
    """
    Applies Augmented Physics-Informed Engineering.
    """
    # 1. Cyclical Augmentation
    df["Aspect_Sin"] = np.sin(np.radians(df["Aspect"]))
    df["Aspect_Cos"] = np.cos(np.radians(df["Aspect"]))

    # 2. Geometric Magnitude
    df["Euclidean_Distance_To_Hydrology"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"] ** 2
        + df["Vertical_Distance_To_Hydrology"] ** 2
    )

    # 3. Directional Preservation
    df["Absolute_Hydrology_Elevation"] = (
        df["Elevation"] - df["Vertical_Distance_To_Hydrology"]
    )

    # 4. Global Context
    amenities = [
        "Horizontal_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Horizontal_Distance_To_Fire_Points",
    ]
    df["Mean_Distance_To_Amenities"] = df[amenities].mean(axis=1)

    return df


def process_data(load_cached_data=True):
    """
    Loads, processes, and caches data.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_files = {
        "X_train": os.path.join(Config.WORKING_DIR, "X_train.npy"),
        "y_train": os.path.join(Config.WORKING_DIR, "y_train.npy"),
        "X_val": os.path.join(Config.WORKING_DIR, "X_val.npy"),
        "y_val": os.path.join(Config.WORKING_DIR, "y_val.npy"),
        "X_test": os.path.join(Config.WORKING_DIR, "X_test.npy"),
        "test_ids": os.path.join(Config.WORKING_DIR, "test_ids.npy"),
    }

    if load_cached_data:
        if all(os.path.exists(p) for p in cache_files.values()):
            print("Loading cached data...")
            return (
                np.load(cache_files["X_train"]),
                np.load(cache_files["y_train"]),
                np.load(cache_files["X_val"]),
                np.load(cache_files["y_val"]),
                np.load(cache_files["X_test"]),
                np.load(cache_files["test_ids"]),
            )

    print("Processing data from scratch...")

    df_train = pd.read_parquet(Config.TRAIN_PATH)
    df_val = pd.read_parquet(Config.VAL_PATH)
    df_test = pd.read_parquet(Config.TEST_PATH)

    df_train = feature_engineering(df_train)
    df_val = feature_engineering(df_val)
    df_test = feature_engineering(df_test)

    target_col = "Cover_Type"
    id_col = "Id"

    y_train = df_train[target_col].values
    y_val = df_val[target_col].values
    test_ids = df_test[id_col].values

    drop_cols = [target_col, id_col]
    X_train_df = df_train.drop(columns=drop_cols, errors="ignore")
    X_val_df = df_val.drop(columns=drop_cols, errors="ignore")
    X_test_df = df_test.drop(columns=[id_col], errors="ignore")

    # Identify continuous and binary columns
    cols = X_train_df.columns
    binary_cols = [
        c for c in cols if c.startswith("Soil_Type") or c.startswith("Wilderness_Area")
    ]
    continuous_cols = [c for c in cols if c not in binary_cols]

    # Standardize continuous features
    scaler = StandardScaler()
    X_train_cont = scaler.fit_transform(
        X_train_df[continuous_cols].values.astype(np.float32)
    )
    X_val_cont = scaler.transform(X_val_df[continuous_cols].values.astype(np.float32))
    X_test_cont = scaler.transform(X_test_df[continuous_cols].values.astype(np.float32))

    # Keep binary features as is
    X_train_bin = X_train_df[binary_cols].values.astype(np.float32)
    X_val_bin = X_val_df[binary_cols].values.astype(np.float32)
    X_test_bin = X_test_df[binary_cols].values.astype(np.float32)

    X_train = np.hstack([X_train_cont, X_train_bin])
    X_val = np.hstack([X_val_cont, X_val_bin])
    X_test = np.hstack([X_test_cont, X_test_bin])

    # Cache results
    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["X_val"], X_val)
    np.save(cache_files["y_val"], y_val)
    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["test_ids"], test_ids)

    return X_train, y_train, X_val, y_val, X_test, test_ids


# ==========================================
# Dataset
# ==========================================
class ForestDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            # Map 1-7 to 0-6
            self.y = torch.tensor(y - 1, dtype=torch.long)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# ==========================================
# Model Architecture
# ==========================================
class VectorDCNLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))
        nn.init.xavier_normal_(self.weight.unsqueeze(0))

    def forward(self, x0, xl):
        # Dot-Product Mixing: x_l^T w -> scalar per sample
        # (xl * weight).sum(dim=1) -> (B, 1)
        dot_prod = (xl * self.weight).sum(dim=1, keepdim=True)
        out = x0 * dot_prod + self.bias + xl
        return out


class PreActResBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear(out)
        return x + out


class DeepParallelVectorDCNResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_blocks, dropout, num_classes):
        super().__init__()

        # Branch 1: Vector DCN (3 layers)
        self.dcn_layers = nn.ModuleList([VectorDCNLayer(input_dim) for _ in range(3)])

        # Branch 2: Pre-Act ResNet
        self.resnet_proj = nn.Linear(input_dim, hidden_dim)
        self.resnet_blocks = nn.ModuleList(
            [PreActResBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )

        # Combination Head
        concat_dim = input_dim + hidden_dim
        self.head = nn.Linear(concat_dim, num_classes)

    def forward(self, x):
        # Branch 1
        x_dcn = x
        for layer in self.dcn_layers:
            x_dcn = layer(x, x_dcn)

        # Branch 2
        x_res = self.resnet_proj(x)
        for block in self.resnet_blocks:
            x_res = block(x_res)

        # Combine
        combined = torch.cat([x_dcn, x_res], dim=1)
        logits = self.head(combined)
        return logits


# ==========================================
# Training & Evaluation
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
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

    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return running_loss / total, correct / total


def predict(model, loader, device):
    model.eval()
    preds = []

    with torch.no_grad():
        for inputs in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            # Map 0-6 back to 1-7
            preds.extend((predicted + 1).cpu().numpy())

    return preds


# ==========================================
# Main Execution
# ==========================================
def main():
    # Load Data
    X_train, y_train, X_val, y_val, X_test, test_ids = process_data(
        load_cached_data=True
    )

    # Datasets & Loaders
    train_dataset = ForestDataset(X_train, y_train)
    val_dataset = ForestDataset(X_val, y_val)
    test_dataset = ForestDataset(X_test, None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model
    input_dim = X_train.shape[1]
    model = DeepParallelVectorDCNResNet(
        input_dim=input_dim,
        hidden_dim=Config.HIDDEN_DIM,
        num_blocks=Config.NUM_BLOCKS,
        dropout=Config.DROPOUT,
        num_classes=Config.NUM_CLASSES,
    ).to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=Config.FACTOR,
        patience=Config.PATIENCE,
        verbose=True,
    )
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0
    early_stop_patience = 10

    for epoch in range(Config.EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} Acc: {train_acc:.6f} | Val Loss: {val_loss:.6f} Acc: {val_acc:.6f}"
        )

        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            print("Early stopping triggered.")
            break

    print(f"Best Validation Accuracy: {best_acc:.6f}")

    # Load best weights
    model.load_state_dict(best_model_wts)

    # Inference
    print("Generating predictions...")
    predictions = predict(model, test_loader, device)

    # Save Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    sub_df = pd.DataFrame({"Id": test_ids, "Cover_Type": predictions})
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
