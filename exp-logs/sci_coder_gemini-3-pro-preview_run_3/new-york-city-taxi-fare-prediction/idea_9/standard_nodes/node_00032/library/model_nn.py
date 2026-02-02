import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error

from library.config import (
    NN_PARAMS,
    WORKING_DIR,
    SEED,
    BB_MIN_LAT,
    BB_MAX_LAT,
    BB_MIN_LON,
    BB_MAX_LON,
    GRID_RESOLUTION,
)
from library.utils import seed_everything, get_device

# ==========================================
# DATASET CLASS
# ==========================================


class TaxiDataset(Dataset):
    def __init__(self, df, mode="train"):
        """
        PyTorch Dataset for Taxi Fare Prediction.

        Args:
            df (pd.DataFrame): Feature-engineered dataframe.
            mode (str): 'train', 'val', or 'test'.
        """
        self.mode = mode
        self.df = df

        # Identify columns
        self.grid_cols = [
            "grid_pickup_lat",
            "grid_pickup_lon",
            "grid_dropoff_lat",
            "grid_dropoff_lon",
        ]
        self.target_col = "fare_amount"
        self.ignore_cols = ["key", "pickup_datetime"]

        # Dense columns are all columns that are not grid, target, or ignored
        all_cols = df.columns.tolist()
        exclude = set(self.grid_cols + [self.target_col] + self.ignore_cols)
        self.dense_cols = [c for c in all_cols if c not in exclude]

        # Pre-convert to numpy arrays for speed
        # Grid indices: (N, 4) int
        self.grid_data = df[self.grid_cols].values.astype(np.int64)

        # Dense features: (N, D) float32
        self.dense_data = df[self.dense_cols].values.astype(np.float32)

        # Target: (N, 1) float32 (only if not test)
        if self.mode != "test" and self.target_col in df.columns:
            self.targets = df[self.target_col].values.astype(np.float32).reshape(-1, 1)
        else:
            self.targets = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        grid = self.grid_data[idx]
        dense = self.dense_data[idx]

        if self.targets is not None:
            target = self.targets[idx]
            return (
                torch.from_numpy(grid),
                torch.from_numpy(dense),
                torch.from_numpy(target),
            )
        else:
            return torch.from_numpy(grid), torch.from_numpy(dense)

    def get_input_dims(self):
        """Returns (vocab_size_lat, vocab_size_lon, n_dense)"""
        # Calculate vocab sizes based on config to ensure consistency across train/test
        # Lat range
        n_lat = (
            int((BB_MAX_LAT - BB_MIN_LAT) / GRID_RESOLUTION) + 2
        )  # +2 for safety/padding
        # Lon range
        n_lon = int((BB_MAX_LON - BB_MIN_LON) / GRID_RESOLUTION) + 2

        # Use the max of both for simplicity in embedding layer definition, or specific
        vocab_size = max(n_lat, n_lon)

        return vocab_size, len(self.dense_cols)


# ==========================================
# MODEL ARCHITECTURE
# ==========================================


class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super(ResBlock, self).__init__()

        # Main path: Linear -> BN -> ReLU -> Dropout
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Shortcut path
        if in_dim != out_dim:
            self.shortcut = nn.Linear(in_dim, out_dim)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        # Main path
        out = self.linear(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Residual Add
        res = self.shortcut(x)
        return out + res


class SpatialResNet(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_dense,
        embedding_dim=64,
        hidden_dims=[512, 256, 128],
        dropout=0.1,
    ):
        super(SpatialResNet, self).__init__()

        # 1. Spatial Embeddings
        # 4 separate embeddings for pickup_lat, pickup_lon, dropoff_lat, dropoff_lon
        self.emb_pickup_lat = nn.Embedding(vocab_size, embedding_dim)
        self.emb_pickup_lon = nn.Embedding(vocab_size, embedding_dim)
        self.emb_dropoff_lat = nn.Embedding(vocab_size, embedding_dim)
        self.emb_dropoff_lon = nn.Embedding(vocab_size, embedding_dim)

        # Total input dimension after concatenating embeddings and dense features
        # 4 embeddings * dim + dense features
        self.in_features = (4 * embedding_dim) + n_dense

        # 2. Residual Backbone
        layers = []
        current_dim = self.in_features

        for h_dim in hidden_dims:
            layers.append(ResBlock(current_dim, h_dim, dropout))
            current_dim = h_dim

        self.backbone = nn.Sequential(*layers)

        # 3. Output Head
        self.head = nn.Linear(current_dim, 1)

    def forward(self, grid_indices, dense_features):
        # grid_indices shape: (Batch, 4) -> [p_lat, p_lon, d_lat, d_lon]

        # Lookup embeddings
        e1 = self.emb_pickup_lat(grid_indices[:, 0])  # (B, Emb)
        e2 = self.emb_pickup_lon(grid_indices[:, 1])  # (B, Emb)
        e3 = self.emb_dropoff_lat(grid_indices[:, 2])  # (B, Emb)
        e4 = self.emb_dropoff_lon(grid_indices[:, 3])  # (B, Emb)

        # Concatenate all features
        x = torch.cat([e1, e2, e3, e4, dense_features], dim=1)

        # Pass through backbone
        x = self.backbone(x)

        # Output
        out = self.head(x)
        return out


# ==========================================
# TRAINING LOGIC
# ==========================================


def train_nn_model(df_train, df_val, load_cached_model=True):
    """
    Trains the Spatial ResNet model.

    Args:
        df_train (pd.DataFrame): Training data.
        df_val (pd.DataFrame): Validation data.
        load_cached_model (bool): Whether to load saved model.

    Returns:
        model (nn.Module): Trained PyTorch model.
    """
    seed_everything(SEED)
    device = get_device()
    model_path = os.path.join(WORKING_DIR, "nn_model.pth")

    # Create Datasets
    train_dataset = TaxiDataset(df_train, mode="train")
    val_dataset = TaxiDataset(df_val, mode="val")

    # Determine dimensions from dataset
    vocab_size, n_dense = train_dataset.get_input_dims()

    # Initialize Model
    model = SpatialResNet(
        vocab_size=vocab_size,
        n_dense=n_dense,
        embedding_dim=NN_PARAMS["embedding_dim"],
        hidden_dims=NN_PARAMS["hidden_dims"],
        dropout=NN_PARAMS["dropout"],
    ).to(device)

    # Load Cached Model if requested
    if load_cached_model and os.path.exists(model_path):
        print(f"Loading Neural Network model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model

    print(f"Training Spatial ResNet (Vocab: {vocab_size}, Dense: {n_dense})...")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=NN_PARAMS["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=NN_PARAMS["batch_size"] * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Optimization
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=NN_PARAMS["learning_rate"],
        weight_decay=NN_PARAMS["weight_decay"],
    )

    # Early Stopping Tracking
    best_val_rmse = float("inf")
    patience_counter = 0
    patience_limit = NN_PARAMS["early_stopping_patience"]

    # Training Loop
    for epoch in range(NN_PARAMS["epochs"]):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0

        for grid, dense, target in train_loader:
            grid = grid.to(device)
            dense = dense.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(grid, dense)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            n_batches += 1

        avg_train_loss = train_loss_sum / n_batches
        avg_train_rmse = np.sqrt(avg_train_loss)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for grid, dense, target in val_loader:
                grid = grid.to(device)
                dense = dense.to(device)
                target = target.to(device)

                output = model(grid, dense)
                loss = criterion(output, target)
                val_loss_sum += loss.item()
                n_val_batches += 1

        avg_val_loss = val_loss_sum / n_val_batches
        avg_val_rmse = np.sqrt(avg_val_loss)

        print(
            f"Epoch {epoch+1}/{NN_PARAMS['epochs']} - Train RMSE: {avg_train_rmse} - Val RMSE: {avg_val_rmse}"
        )

        # Early Stopping Check
        if avg_val_rmse < best_val_rmse:
            best_val_rmse = avg_val_rmse
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Best Validation RMSE: {best_val_rmse}")
    print(f"Saving Neural Network model to {model_path}...")

    # Reload best model
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def predict_nn_model(model, df):
    """
    Generates predictions using the Spatial ResNet.

    Args:
        model: Trained PyTorch model.
        df (pd.DataFrame): Data to predict on.

    Returns:
        np.array: Predictions (flattened).
    """
    device = get_device()
    model.eval()
    model.to(device)

    dataset = TaxiDataset(df, mode="test")
    loader = DataLoader(
        dataset,
        batch_size=NN_PARAMS["batch_size"] * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    predictions = []

    with torch.no_grad():
        for batch in loader:
            # Handle case where dataset returns 2 or 3 items
            if len(batch) == 3:
                grid, dense, _ = batch
            else:
                grid, dense = batch

            grid = grid.to(device)
            dense = dense.to(device)

            output = model(grid, dense)
            predictions.append(output.cpu().numpy())

    return np.concatenate(predictions).flatten()
