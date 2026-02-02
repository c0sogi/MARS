import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import pandas as pd
import os
import math
from sklearn.preprocessing import StandardScaler

# Import from library
from library.config import (
    SEED,
    METADATA_DIR,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    WORKING_DIR,
    TRAIN_CACHE_PATH,
    TRAIN_LABELS_PATH,
    VAL_CACHE_PATH,
    VAL_LABELS_PATH,
    TEST_CACHE_PATH,
    TEST_IDS_PATH,
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PATIENCE,
    NUM_WORKERS,
    HIDDEN_DIM,
    NUM_CROSS_LAYERS,
    SE_REDUCTION_RATIO,
    DROPOUT_RATE,
    NUM_CLASSES,
)
from library.utils import seed_everything, ModelCheckpoint

# =============================================================================
# Model Components
# =============================================================================


class CrossNetVector(nn.Module):
    """
    Vector-based Cross Network (DCN v1 style formulation).
    Formula: x_{l+1} = x_0 * (x_l^T w) + b + x_l
    This performs explicit feature interaction mixing using a learned scalar projection.
    """

    def __init__(self, input_dim, num_layers):
        super(CrossNetVector, self).__init__()
        self.num_layers = num_layers
        # Parameters: w and b for each layer
        # w: (num_layers, input_dim)
        # b: (num_layers, input_dim)
        self.w = nn.Parameter(torch.randn(num_layers, input_dim))
        self.b = nn.Parameter(torch.zeros(num_layers, input_dim))

        # Initialize weights
        nn.init.xavier_uniform_(self.w)

    def forward(self, x0):
        # x0: (Batch, Input_Dim)
        xl = x0
        for i in range(self.num_layers):
            # Term: x_l^T w
            # x_l: (Batch, D), w[i]: (D)
            # Dot product for each sample: (Batch, 1)
            w_i = self.w[i]  # (D)
            b_i = self.b[i]  # (D)

            # Efficient dot product computation
            dot_prod = (xl * w_i).sum(dim=1, keepdim=True)  # (Batch, 1)

            # Update formula: x_{l+1} = x_0 * dot_prod + b + x_l
            xl = x0 * dot_prod + b_i + xl

        return xl


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Residual Block for 1D tabular data.
    Structure:
    Input -> Dense -> BN -> ReLU -> Dropout -> Dense -> BN -> SE -> Add Input -> ReLU
    """

    def __init__(self, input_dim, hidden_dim, reduction_ratio, dropout_rate):
        super(SEBlock, self).__init__()

        # Main path
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # SE Module (Excitation)
        # For 1D vectors, the "Squeeze" (Global Average Pooling) is implicit/identity.
        # We model channel-wise dependencies directly.
        reduced_dim = max(1, hidden_dim // reduction_ratio)
        self.se_fc1 = nn.Linear(hidden_dim, reduced_dim)
        self.se_act = nn.ReLU()
        self.se_fc2 = nn.Linear(reduced_dim, hidden_dim)
        self.se_sigmoid = nn.Sigmoid()

        # Shortcut handling for residual connection
        if input_dim != hidden_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim)
            )
        else:
            self.shortcut = nn.Identity()

        self.final_act = nn.ReLU()

    def forward(self, x):
        # Main Path
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn2(out)

        # SE Path (Attention)
        se = self.se_fc1(out)
        se = self.se_act(se)
        se = self.se_fc2(se)
        se_weights = self.se_sigmoid(se)  # (Batch, Hidden)

        # Scale Output
        out = out * se_weights

        # Residual connection
        res = self.shortcut(x)
        out = out + res

        out = self.final_act(out)
        return out


class ParallelDCNSEResNet(nn.Module):
    """
    Hybrid architecture combining a Vector-DCN branch for explicit interactions
    and an SE-ResNet branch for deep implicit feature learning.
    """

    def __init__(
        self,
        input_dim,
        num_classes,
        hidden_dim,
        num_cross_layers,
        se_reduction,
        dropout,
    ):
        super(ParallelDCNSEResNet, self).__init__()

        # Branch 1: Vector DCN (Explicit Interactions)
        # Keeps dimension same as input_dim throughout
        self.dcn = CrossNetVector(input_dim, num_cross_layers)

        # Branch 2: SE-ResNet (Deep Implicit Features)
        # Project to hidden space then apply residual blocks
        self.resnet_input_proj = nn.Linear(input_dim, hidden_dim)
        self.resnet_blocks = nn.Sequential(
            SEBlock(hidden_dim, hidden_dim, se_reduction, dropout),
            SEBlock(hidden_dim, hidden_dim, se_reduction, dropout),
        )

        # Combination Head
        # Concatenate DCN output (input_dim) and ResNet output (hidden_dim)
        concat_dim = input_dim + hidden_dim

        self.head = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        # x: (Batch, Input_Dim)

        # Branch 1
        x_dcn = self.dcn(x)

        # Branch 2
        x_res = self.resnet_input_proj(x)
        x_res = self.resnet_blocks(x_res)

        # Combine
        x_concat = torch.cat([x_dcn, x_res], dim=1)

        logits = self.head(x_concat)
        return logits


# =============================================================================
# Data Processing
# =============================================================================


def feature_engineering(df):
    """
    Applies physics-informed feature augmentation.
    """
    # 1. Cyclical Aspect (Preserve raw Aspect as well)
    df["Aspect_Sin"] = np.sin(df["Aspect"] * np.pi / 180.0)
    df["Aspect_Cos"] = np.cos(df["Aspect"] * np.pi / 180.0)

    # 2. Euclidean Distance to Hydrology (Hypotenuse)
    df["Hydrology_Distance_Euclidean"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"] ** 2
        + df["Vertical_Distance_To_Hydrology"] ** 2
    )

    # 3. Absolute Hydrology Elevation (Directional preservation)
    # Vertical_Dist = Elev - Hydro_Elev -> Hydro_Elev = Elev - Vertical_Dist
    df["Hydrology_Elevation_Abs"] = (
        df["Elevation"] - df["Vertical_Distance_To_Hydrology"]
    )

    # 4. Amenities Mean Distance (Global context shortcut)
    df["Amenities_Mean_Dist"] = (
        df["Horizontal_Distance_To_Hydrology"]
        + df["Horizontal_Distance_To_Roadways"]
        + df["Horizontal_Distance_To_Fire_Points"]
    ) / 3.0

    return df


def get_processed_data(load_cached_data=True):
    """
    Loads, processes, and caches data.
    Returns: (train_X, train_y, val_X, val_y, test_X, test_ids, input_dim)
    """

    # Check if cache exists
    if (
        load_cached_data
        and os.path.exists(TRAIN_CACHE_PATH)
        and os.path.exists(TRAIN_LABELS_PATH)
        and os.path.exists(VAL_CACHE_PATH)
        and os.path.exists(VAL_LABELS_PATH)
        and os.path.exists(TEST_CACHE_PATH)
        and os.path.exists(TEST_IDS_PATH)
    ):

        print("Loading cached data...")
        train_X = np.load(TRAIN_CACHE_PATH)
        train_y = np.load(TRAIN_LABELS_PATH)
        val_X = np.load(VAL_CACHE_PATH)
        val_y = np.load(VAL_LABELS_PATH)
        test_X = np.load(TEST_CACHE_PATH)
        test_ids = np.load(TEST_IDS_PATH)

        return train_X, train_y, val_X, val_y, test_X, test_ids, train_X.shape[1]

    print("Processing data from scratch...")

    # Load Parquet
    df_train = pd.read_parquet(TRAIN_DATA_PATH)
    df_val = pd.read_parquet(VAL_DATA_PATH)
    df_test = pd.read_parquet(TEST_DATA_PATH)

    # Extract IDs and Targets
    train_y = df_train["Cover_Type"].values
    val_y = df_val["Cover_Type"].values
    test_ids = df_test["Id"].values

    # Drop Id and Target from features
    drop_cols_train = ["Id", "Cover_Type"]
    drop_cols_test = ["Id"]

    X_train_df = df_train.drop(columns=drop_cols_train)
    X_val_df = df_val.drop(columns=drop_cols_train)
    X_test_df = df_test.drop(columns=drop_cols_test)

    # Feature Engineering
    X_train_df = feature_engineering(X_train_df)
    X_val_df = feature_engineering(X_val_df)
    X_test_df = feature_engineering(X_test_df)

    # Identify Columns
    # Binary columns: Soil_Type* and Wilderness_Area*
    binary_cols = [
        c
        for c in X_train_df.columns
        if c.startswith("Soil_Type") or c.startswith("Wilderness_Area")
    ]
    # Continuous columns: All others
    continuous_cols = [c for c in X_train_df.columns if c not in binary_cols]

    # Standardization (Continuous only)
    scaler = StandardScaler()

    # Fit on Train, Transform all
    X_train_df[continuous_cols] = scaler.fit_transform(
        X_train_df[continuous_cols].astype(np.float32)
    )
    X_val_df[continuous_cols] = scaler.transform(
        X_val_df[continuous_cols].astype(np.float32)
    )
    X_test_df[continuous_cols] = scaler.transform(
        X_test_df[continuous_cols].astype(np.float32)
    )

    # Convert to Numpy (float32)
    # Ensure column order is identical
    all_cols = continuous_cols + binary_cols

    train_X = X_train_df[all_cols].values.astype(np.float32)
    val_X = X_val_df[all_cols].values.astype(np.float32)
    test_X = X_test_df[all_cols].values.astype(np.float32)

    # Adjust Targets to 0-indexed (1-7 -> 0-6)
    train_y = (train_y - 1).astype(np.int64)
    val_y = (val_y - 1).astype(np.int64)

    # Cache
    np.save(TRAIN_CACHE_PATH, train_X)
    np.save(TRAIN_LABELS_PATH, train_y)
    np.save(VAL_CACHE_PATH, val_X)
    np.save(VAL_LABELS_PATH, val_y)
    np.save(TEST_CACHE_PATH, test_X)
    np.save(TEST_IDS_PATH, test_ids)

    return train_X, train_y, val_X, val_y, test_X, test_ids, train_X.shape[1]


# =============================================================================
# Training & Inference
# =============================================================================


def train_model(load_cached_data=True):
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Data
    train_X, train_y, val_X, val_y, test_X, test_ids, input_dim = get_processed_data(
        load_cached_data
    )

    # Datasets & Loaders
    train_dataset = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y))
    val_dataset = TensorDataset(torch.from_numpy(val_X), torch.from_numpy(val_y))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Model
    model = ParallelDCNResNet(
        input_dim=input_dim,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_cross_layers=NUM_CROSS_LAYERS,
        dropout=DROPOUT_RATE,
    ).to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=3
    )
    criterion = nn.CrossEntropyLoss()

    # Checkpoint
    checkpoint = ModelCheckpoint(mode="max")

    print(f"Starting training on {device}...")

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

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
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.6f} | Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.6f}"
        )

        # Scheduler Step
        scheduler.step(val_acc)

        # Checkpoint & Early Stopping
        improved = checkpoint.step(val_acc, model)
        if improved:
            best_val_acc = val_acc
            patience_counter = 0
            checkpoint.save_best(MODEL_SAVE_PATH)
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training complete. Best Validation Accuracy: {best_val_acc:.6f}")

    # Generate Submission
    predict_and_submit(model, test_X, test_ids, device, checkpoint)


def predict_and_submit(model, test_X, test_ids, device, checkpoint):
    print("Generating predictions...")

    # Load best weights
    model = checkpoint.load_best(model)
    model.eval()

    test_dataset = TensorDataset(torch.from_numpy(test_X))
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    predictions = []

    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch[0].to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            # Map back to 1-7 (add 1)
            predicted = predicted + 1
            predictions.extend(predicted.cpu().numpy())

    # Create submission dataframe
    df_sub = pd.DataFrame({"Id": test_ids, "Cover_Type": predictions})

    df_sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
