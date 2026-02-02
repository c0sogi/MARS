import os
import gc
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


# ==========================================
# Configuration
# ==========================================
class Config:
    # Data Paths
    METADATA_DIR = "./metadata"
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Cache
    CACHE_DIR = "./working/idea_22"

    # Output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model Hyperparameters
    SEED = 42
    BATCH_SIZE = 4096
    EPOCHS = 60
    LEARNING_RATE = 1e-3
    DROPOUT = 0.2
    RESNET_WIDTH = 512
    RESNET_DEPTH = 4
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 3
    EARLY_STOPPING_PATIENCE = 8

    # Data Params
    NUM_CLASSES = 7  # Cover_Type 1-7

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4


# ==========================================
# Utils
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================
# Data Processing
# ==========================================
def feature_engineering(df):
    """
    Applies the augmented physics-informed engineering.
    """
    df = df.copy()

    # 1. Cyclical Augmentation (Keep raw Aspect)
    df["Aspect_Sin"] = np.sin(np.radians(df["Aspect"]))
    df["Aspect_Cos"] = np.cos(np.radians(df["Aspect"]))

    # 2. Geometric Magnitude (Euclidean Distance to Hydrology)
    h_dist = df["Horizontal_Distance_To_Hydrology"]
    v_dist = df["Vertical_Distance_To_Hydrology"]
    df["Euclidean_Dist_Hydro"] = np.sqrt(h_dist**2 + v_dist**2)

    # 3. Directional Preservation (Absolute Hydrology Elevation)
    df["Abs_Hydro_Elev"] = df["Elevation"] - v_dist

    # 4. Global Context (Mean Distance to Amenities)
    d_hydro = df["Horizontal_Distance_To_Hydrology"].abs()
    d_road = df["Horizontal_Distance_To_Roadways"].abs()
    d_fire = df["Horizontal_Distance_To_Fire_Points"].abs()
    df["Mean_Dist_Amenities"] = (d_hydro + d_road + d_fire) / 3.0

    return df


def get_data(load_cached_data=True):
    """
    Loads, processes, and caches data.
    Returns: (X_train, y_train), (X_val, y_val), (X_test, test_ids), scaler
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    cache_files = {
        "train_X": os.path.join(Config.CACHE_DIR, "train_X.npy"),
        "train_y": os.path.join(Config.CACHE_DIR, "train_y.npy"),
        "val_X": os.path.join(Config.CACHE_DIR, "val_X.npy"),
        "val_y": os.path.join(Config.CACHE_DIR, "val_y.npy"),
        "test_X": os.path.join(Config.CACHE_DIR, "test_X.npy"),
        "test_ids": os.path.join(Config.CACHE_DIR, "test_ids.npy"),
        "scaler_mean": os.path.join(Config.CACHE_DIR, "scaler_mean.npy"),
        "scaler_scale": os.path.join(Config.CACHE_DIR, "scaler_scale.npy"),
    }

    # Check cache
    if load_cached_data:
        all_exist = all(os.path.exists(p) for p in cache_files.values())
        if all_exist:
            print("Loading cached data...")
            X_train = np.load(cache_files["train_X"])
            y_train = np.load(cache_files["train_y"])
            X_val = np.load(cache_files["val_X"])
            y_val = np.load(cache_files["val_y"])
            X_test = np.load(cache_files["test_X"])
            test_ids = np.load(cache_files["test_ids"])

            # Reconstruct scaler
            scaler = StandardScaler()
            scaler.mean_ = np.load(cache_files["scaler_mean"])
            scaler.scale_ = np.load(cache_files["scaler_scale"])
            scaler.var_ = scaler.scale_**2
            scaler.n_samples_seen_ = len(X_train)

            return (X_train, y_train), (X_val, y_val), (X_test, test_ids), scaler

    print("Processing data from scratch...")

    # Load Parquet
    df_train = pd.read_parquet(Config.TRAIN_PATH)
    df_val = pd.read_parquet(Config.VAL_PATH)
    df_test = pd.read_parquet(Config.TEST_PATH)

    # Feature Engineering
    df_train = feature_engineering(df_train)
    df_val = feature_engineering(df_val)
    df_test = feature_engineering(df_test)

    # Identify Columns
    target_col = "Cover_Type"
    id_col = "Id"

    # Binary Columns (Wilderness_Area, Soil_Type)
    bin_cols = [
        c
        for c in df_train.columns
        if c.startswith("Wilderness_Area") or c.startswith("Soil_Type")
    ]

    # Continuous Columns (All others except Id and Target)
    exclude = [target_col, id_col] + bin_cols
    cont_cols = [c for c in df_train.columns if c not in exclude]

    # Prepare Arrays
    y_train = df_train[target_col].values - 1  # 0-indexed
    y_val = df_val[target_col].values - 1
    test_ids = df_test[id_col].values

    X_train_cont = df_train[cont_cols].values.astype(np.float32)
    X_val_cont = df_val[cont_cols].values.astype(np.float32)
    X_test_cont = df_test[cont_cols].values.astype(np.float32)

    X_train_bin = df_train[bin_cols].values.astype(np.float32)
    X_val_bin = df_val[bin_cols].values.astype(np.float32)
    X_test_bin = df_test[bin_cols].values.astype(np.float32)

    # Standardization (Continuous only)
    scaler = StandardScaler()
    X_train_cont = scaler.fit_transform(X_train_cont)
    X_val_cont = scaler.transform(X_val_cont)
    X_test_cont = scaler.transform(X_test_cont)

    # Concatenate
    X_train = np.hstack([X_train_cont, X_train_bin])
    X_val = np.hstack([X_val_cont, X_val_bin])
    X_test = np.hstack([X_test_cont, X_test_bin])

    # Save to cache
    np.save(cache_files["train_X"], X_train)
    np.save(cache_files["train_y"], y_train)
    np.save(cache_files["val_X"], X_val)
    np.save(cache_files["val_y"], y_val)
    np.save(cache_files["test_X"], X_test)
    np.save(cache_files["test_ids"], test_ids)
    np.save(cache_files["scaler_mean"], scaler.mean_)
    np.save(cache_files["scaler_scale"], scaler.scale_)

    return (X_train, y_train), (X_val, y_val), (X_test, test_ids), scaler


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


# ==========================================
# Model Architecture
# ==========================================
class VectorCrossLayer(nn.Module):
    """
    Vector-based (Rank-1) Cross Layer with Dot-Product Mixing.
    x_{l+1} = x_0 * (x_l . w) + b + x_l
    """

    def __init__(self, input_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(input_dim))
        self.bias = nn.Parameter(torch.Tensor(input_dim))
        nn.init.normal_(self.weight, std=0.01)
        nn.init.zeros_(self.bias)

    def forward(self, x0, xl):
        # Dot product (xl . w) -> scalar per sample
        dot_prod = torch.sum(xl * self.weight, dim=1, keepdim=True)
        # Mix
        out = x0 * dot_prod + self.bias + xl
        return out


class ResNetBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class DeepParallelNet(nn.Module):
    def __init__(
        self, input_dim, num_classes, resnet_depth=4, resnet_width=512, dropout=0.2
    ):
        super().__init__()

        # Branch 1: Vector DCN
        self.num_cross_layers = 4
        self.cross_layers = nn.ModuleList(
            [VectorCrossLayer(input_dim) for _ in range(self.num_cross_layers)]
        )

        # Branch 2: Deep ResNet
        self.resnet_input = nn.Sequential(
            nn.Linear(input_dim, resnet_width), nn.ReLU(), nn.Dropout(dropout)
        )
        self.resnet_blocks = nn.ModuleList(
            [ResNetBlock(resnet_width, dropout) for _ in range(resnet_depth)]
        )

        # Combination Head
        concat_dim = input_dim + resnet_width
        self.head = nn.Linear(concat_dim, num_classes)

    def forward(self, x):
        # Branch 1: DCN
        x0 = x
        xl = x
        for layer in self.cross_layers:
            xl = layer(x0, xl)
        dcn_out = xl

        # Branch 2: ResNet
        res = self.resnet_input(x)
        for block in self.resnet_blocks:
            res = block(res)
        resnet_out = res

        # Combine
        combined = torch.cat([dcn_out, resnet_out], dim=1)
        logits = self.head(combined)
        return logits


# ==========================================
# Training & Inference
# ==========================================
def train_model(X_train, y_train, X_val, y_val):
    set_seed(Config.SEED)

    train_ds = CoverTypeDataset(X_train, y_train)
    val_ds = CoverTypeDataset(X_val, y_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    model = DeepParallelNet(
        input_dim=X_train.shape[1],
        num_classes=Config.NUM_CLASSES,
        resnet_depth=Config.RESNET_DEPTH,
        resnet_width=Config.RESNET_WIDTH,
        dropout=Config.DROPOUT,
    ).to(Config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        verbose=True,
    )

    best_acc = 0.0
    best_model_state = None
    patience_counter = 0

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

        train_acc = correct / total
        avg_train_loss = train_loss / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(Config.DEVICE), y_batch.to(Config.DEVICE)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()

        val_acc = val_correct / val_total
        avg_val_loss = val_loss / val_total

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {avg_train_loss:.6f} | Train Acc: {train_acc:.6f} | "
            f"Val Loss: {avg_val_loss:.6f} | Val Acc: {val_acc:.6f}"
        )

        # Scheduler step
        scheduler.step(val_acc)

        # Early Stopping & Checkpointing
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def predict_and_submit(model, X_test, test_ids):
    model.eval()
    test_ds = CoverTypeDataset(X_test)
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    predictions = []

    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(Config.DEVICE)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            # Map back to 1-7 (add 1)
            preds = predicted.cpu().numpy() + 1
            predictions.extend(preds)

    # Save submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_sub = pd.DataFrame({"Id": test_ids, "Cover_Type": predictions})
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_pipeline():
    # 1. Load Data
    (X_train, y_train), (X_val, y_val), (X_test, test_ids), scaler = get_data(
        load_cached_data=True
    )

    # 2. Train
    model = train_model(X_train, y_train, X_val, y_val)

    # 3. Predict
    predict_and_submit(model, X_test, test_ids)
