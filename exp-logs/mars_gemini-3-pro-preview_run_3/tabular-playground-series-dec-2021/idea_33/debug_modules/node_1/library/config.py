import os
import sys
import gc
import copy
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ==================================================================================
# CONFIGURATION & HYPERPARAMETERS
# ==================================================================================


class Config:
    # Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_33")
    SUBMISSION_DIR = "./submission"

    # Files
    TRAIN_META = os.path.join(METADATA_DIR, "train.parquet")
    VAL_META = os.path.join(METADATA_DIR, "val.parquet")
    TEST_META = os.path.join(METADATA_DIR, "test.parquet")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Data
    TARGET_COL = "Cover_Type"
    ID_COL = "Id"
    SEED = 42

    # Model Architecture
    HIDDEN_DIM = 512
    RESNET_BLOCKS = 6
    DCN_LAYERS = 3
    DROPOUT = 0.2
    NUM_CLASSES = 7  # Classes 1-7

    # Training
    BATCH_SIZE = 4096
    EPOCHS = 60
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    PATIENCE = 5
    FACTOR = 0.1
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Ensure directories exist
os.makedirs(Config.CACHE_DIR, exist_ok=True)
os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)


# Set Random Seeds
def set_seed(seed=Config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Disable strict determinism for performance as per Lesson 00070
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


set_seed()

# ==================================================================================
# DATA PROCESSING
# ==================================================================================


def feature_engineering(df):
    """
    Applies Augmented Physics-Informed Engineering.
    """
    # 1. Cyclical Augmentation (Keep raw Aspect)
    df["Aspect_Sin"] = np.sin(df["Aspect"] * np.pi / 180.0)
    df["Aspect_Cos"] = np.cos(df["Aspect"] * np.pi / 180.0)

    # 2. Geometric Magnitude (Euclidean Distance to Hydrology)
    # H^2 + V^2
    df["Hydrology_Dist"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"] ** 2
        + df["Vertical_Distance_To_Hydrology"] ** 2
    )

    # 3. Directional Preservation (Absolute Hydrology Elevation)
    df["Abs_Hydro_Elev"] = df["Elevation"] - df["Vertical_Distance_To_Hydrology"]

    # 4. Global Context (Mean Distance to Amenities)
    # Hydro, Road, Fire
    df["Mean_Amenities"] = (
        df["Horizontal_Distance_To_Hydrology"]
        + df["Horizontal_Distance_To_Roadways"]
        + df["Horizontal_Distance_To_Fire_Points"]
    ) / 3.0

    return df


def get_feature_groups(columns):
    """
    Identifies continuous and binary feature columns.
    """
    # Binary features are Soil_TypeX and Wilderness_AreaX
    binary_cols = [
        c
        for c in columns
        if c.startswith("Soil_Type") or c.startswith("Wilderness_Area")
    ]

    # Continuous are the rest (excluding ID and Target)
    exclude = binary_cols + [Config.ID_COL, Config.TARGET_COL]
    continuous_cols = [c for c in columns if c not in exclude]

    return continuous_cols, binary_cols


def process_data(load_cached_data=True):
    """
    Loads, processes, and caches data.
    """
    cache_train_x = os.path.join(Config.CACHE_DIR, "train_X.npy")
    cache_train_y = os.path.join(Config.CACHE_DIR, "train_y.npy")
    cache_val_x = os.path.join(Config.CACHE_DIR, "val_X.npy")
    cache_val_y = os.path.join(Config.CACHE_DIR, "val_y.npy")
    cache_test_x = os.path.join(Config.CACHE_DIR, "test_X.npy")
    cache_test_ids = os.path.join(Config.CACHE_DIR, "test_ids.npy")

    if load_cached_data and os.path.exists(cache_train_x):
        print("Loading cached data...")
        X_train = np.load(cache_train_x)
        y_train = np.load(cache_train_y)
        X_val = np.load(cache_val_x)
        y_val = np.load(cache_val_y)
        X_test = np.load(cache_test_x)
        test_ids = np.load(cache_test_ids)
        return X_train, y_train, X_val, y_val, X_test, test_ids

    print("Processing data from scratch...")

    # Load Metadata Parquets
    df_train = pd.read_parquet(Config.TRAIN_META)
    df_val = pd.read_parquet(Config.VAL_META)
    df_test = pd.read_parquet(Config.TEST_META)

    # Feature Engineering
    df_train = feature_engineering(df_train)
    df_val = feature_engineering(df_val)
    df_test = feature_engineering(df_test)

    # Identify Columns
    cont_cols, bin_cols = get_feature_groups(df_train.columns)

    # Standardization (Fit on Train, Transform All)
    scaler = StandardScaler()
    df_train[cont_cols] = scaler.fit_transform(df_train[cont_cols].astype(np.float32))
    df_val[cont_cols] = scaler.transform(df_val[cont_cols].astype(np.float32))
    df_test[cont_cols] = scaler.transform(df_test[cont_cols].astype(np.float32))

    # Prepare Arrays
    # Concatenate Continuous and Binary
    feature_cols = cont_cols + bin_cols

    X_train = df_train[feature_cols].values.astype(np.float32)
    y_train = (df_train[Config.TARGET_COL].values - 1).astype(np.int64)  # 0-indexed

    X_val = df_val[feature_cols].values.astype(np.float32)
    y_val = (df_val[Config.TARGET_COL].values - 1).astype(np.int64)

    X_test = df_test[feature_cols].values.astype(np.float32)
    test_ids = df_test[Config.ID_COL].values.astype(np.int64)

    # Cache Data
    np.save(cache_train_x, X_train)
    np.save(cache_train_y, y_train)
    np.save(cache_val_x, X_val)
    np.save(cache_val_y, y_val)
    np.save(cache_test_x, X_test)
    np.save(cache_test_ids, test_ids)

    return X_train, y_train, X_val, y_val, X_test, test_ids


class CoverTypeDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# ==================================================================================
# MODEL ARCHITECTURE
# ==================================================================================


class VectorCrossLayer(nn.Module):
    """
    Vector-based (Rank-1) Cross Layer with Dot-Product Mixing.
    Formula: x_{l+1} = x_0 * (x_l . w) + b + x_l
    where (.) is dot product, resulting in a scalar gating of x_0.
    """

    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.w = nn.Parameter(torch.empty(input_dim))
        self.b = nn.Parameter(torch.empty(input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        # Initialization: Near-Zero Standard Deviation
        nn.init.normal_(self.w, mean=0, std=1e-4)
        nn.init.zeros_(self.b)

    def forward(self, x0, xl):
        # x0: [batch, dim], xl: [batch, dim], w: [dim]
        # interaction = (xl * w).sum(dim=1) -> [batch] (Scalar per sample)
        interaction = (xl * self.w).sum(dim=1, keepdim=True)
        return x0 * interaction + self.b + xl


class PreActResNetBlock(nn.Module):
    """
    Full Pre-Activation ResNet Block.
    BN -> ReLU -> Dropout -> Linear -> BN -> ReLU -> Dropout -> Linear -> Add
    """

    def __init__(self, dim, dropout):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(dim)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.lin1 = nn.Linear(dim, dim)

        self.bn2 = nn.BatchNorm1d(dim)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.lin2 = nn.Linear(dim, dim)

        self.reset_parameters()

    def reset_parameters(self):
        # Standard init for Linear
        nn.init.kaiming_normal_(self.lin1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.lin2.weight, nonlinearity="relu")
        nn.init.zeros_(self.lin1.bias)
        nn.init.zeros_(self.lin2.bias)

        # Zero-Gamma Initialization for the second BN
        nn.init.ones_(self.bn1.weight)
        nn.init.zeros_(self.bn1.bias)
        nn.init.constant_(self.bn2.weight, 0.0)  # Zero Init
        nn.init.zeros_(self.bn2.bias)

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.act1(out)
        out = self.drop1(out)
        out = self.lin1(out)

        out = self.bn2(out)
        out = self.act2(out)
        out = self.drop2(out)
        out = self.lin2(out)

        return out + residual


class ZeroInitDeepAsymmetricNet(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, num_blocks, dcn_layers, num_classes, dropout
    ):
        super().__init__()

        # Input Projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Branch 1: Vector DCN (Warm-Start)
        self.dcn_layers = nn.ModuleList(
            [VectorCrossLayer(hidden_dim) for _ in range(dcn_layers)]
        )

        # Branch 2: Deep Pre-Act ResNet (Zero-Init)
        self.resnet_blocks = nn.ModuleList(
            [PreActResNetBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )

        # Head
        # Concatenation of both branches (hidden_dim * 2) -> Output
        self.head = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # Project input to hidden dim
        x_proj = self.input_proj(x)

        # Branch 1: DCN
        x_dcn = x_proj
        for layer in self.dcn_layers:
            x_dcn = layer(x_proj, x_dcn)

        # Branch 2: ResNet
        x_res = x_proj
        for block in self.resnet_blocks:
            x_res = block(x_res)

        # Combine
        combined = torch.cat([x_dcn, x_res], dim=1)
        logits = self.head(combined)
        return logits


# ==================================================================================
# TRAINING LOOP
# ==================================================================================


def train_model():
    # Load Data
    X_train, y_train, X_val, y_val, X_test, test_ids = process_data(
        load_cached_data=True
    )

    train_dataset = CoverTypeDataset(X_train, y_train)
    val_dataset = CoverTypeDataset(X_val, y_val)

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

    # Initialize Model
    model = ZeroInitDeepAsymmetricNet(
        input_dim=X_train.shape[1],
        hidden_dim=Config.HIDDEN_DIM,
        num_blocks=Config.RESNET_BLOCKS,
        dcn_layers=Config.DCN_LAYERS,
        num_classes=Config.NUM_CLASSES,
        dropout=Config.DROPOUT,
    ).to(Config.DEVICE)

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
    best_model_state = None
    no_improve_epochs = 0

    print(f"Starting training on {Config.DEVICE}...")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(Config.DEVICE), y_batch.to(Config.DEVICE)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        train_loss /= total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(Config.DEVICE), y_batch.to(Config.DEVICE)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total

        # Scheduler Step
        scheduler.step(val_acc)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.6f}"
        )

        # Early Stopping & Checkpointing
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if (
            no_improve_epochs >= Config.PATIENCE * 2
        ):  # Give a bit more room than scheduler
            print("Early stopping triggered.")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with Val Acc: {best_acc:.6f}")

    return model, X_test, test_ids


def generate_submission(model, X_test, test_ids):
    model.eval()
    test_dataset = CoverTypeDataset(X_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    predictions = []

    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(Config.DEVICE)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            # Map 0-6 back to 1-7
            predicted = predicted + 1
            predictions.extend(predicted.cpu().numpy())

    df_sub = pd.DataFrame({Config.ID_COL: test_ids, Config.TARGET_COL: predictions})

    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


# ==================================================================================
# MAIN EXECUTION
# ==================================================================================

if __name__ == "__main__":
    model, X_test, test_ids = train_model()
    generate_submission(model, X_test, test_ids)
