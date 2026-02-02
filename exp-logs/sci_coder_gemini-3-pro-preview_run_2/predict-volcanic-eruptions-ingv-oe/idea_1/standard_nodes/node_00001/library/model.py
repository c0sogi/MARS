import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import random
from library.config import Config
from library.feature_extractor import extract_features


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(Config.SEED)


# ---------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------
class VolcanoMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers=None, dropout_rate=0.3):
        """
        Multi-Layer Perceptron for Seismic Regression.

        Args:
            input_dim (int): Number of input features.
            hidden_layers (list): List of hidden layer sizes.
            dropout_rate (float): Dropout probability.
        """
        super(VolcanoMLP, self).__init__()
        if hidden_layers is None:
            hidden_layers = [256, 128, 64]

        layers = []
        in_dim = input_dim

        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim

        # Output layer (Scalar regression)
        layers.append(nn.Linear(in_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ---------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------
def prepare_data(debug_size=None):
    """
    Loads features, scales data, and returns DataLoaders.
    Saves scaler parameters for inference.
    """
    # Load features (uses caching internally)
    df_train = extract_features(
        Config.TRAIN_METADATA_PATH, Config.TRAIN_FEATURES_CACHE, debug_size=debug_size
    )
    df_val = extract_features(
        Config.VAL_METADATA_PATH, Config.VAL_FEATURES_CACHE, debug_size=debug_size
    )

    # Identify feature columns (exclude metadata/target)
    feature_cols = [
        c for c in df_train.columns if c not in ["segment_id", "time_to_eruption"]
    ]

    # Prepare Arrays
    X_train = df_train[feature_cols].values.astype(np.float32)
    y_train = df_train["time_to_eruption"].values.astype(np.float32)

    X_val = df_val[feature_cols].values.astype(np.float32)
    y_val = df_val["time_to_eruption"].values.astype(np.float32)

    # Scaling (Standardization)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Save Scaler Parameters (Avoid pickle, save raw arrays)
    mean_path = os.path.join(Config.WORKING_DIR, "scaler_mean.npy")
    scale_path = os.path.join(Config.WORKING_DIR, "scaler_scale.npy")
    np.save(mean_path, scaler.mean_)
    np.save(scale_path, scaler.scale_)

    # Create TensorDatasets
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    return train_loader, val_loader, len(feature_cols), feature_cols


# ---------------------------------------------------------
# Training Logic
# ---------------------------------------------------------
def train_model(
    train_loader,
    val_loader,
    input_dim,
    device,
    epochs=100,
    lr=1e-3,
    patience=10,
    model_save_path=None,
):
    """
    Trains the VolcanoMLP model with early stopping.
    """
    model = VolcanoMLP(
        input_dim=input_dim,
        hidden_layers=Config.HIDDEN_LAYERS,
        dropout_rate=Config.DROPOUT_RATE,
    ).to(device)

    criterion = nn.L1Loss()  # MAE Loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss_sum = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * X_batch.size(0)

        avg_train_loss = train_loss_sum / len(train_loader.dataset)

        # --- Validation ---
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                val_loss_sum += loss.item() * X_batch.size(0)

        avg_val_loss = val_loss_sum / len(val_loader.dataset)

        # --- Updates ---
        scheduler.step(avg_val_loss)

        print(
            f"Epoch {epoch+1}/{epochs} - Train MAE: {avg_train_loss} - Val MAE: {avg_val_loss}"
        )

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            if model_save_path:
                torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Best Validation MAE: {best_val_loss}")

    # Reload best model state
    if model_save_path and os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))

    return model


# ---------------------------------------------------------
# Inference & Submission
# ---------------------------------------------------------
def generate_submission(model, feature_cols, device, debug_size=None):
    """
    Generates predictions for the test set and saves submission.csv.
    """
    print("Generating submission...")
    df_test = extract_features(
        Config.TEST_METADATA_PATH, Config.TEST_FEATURES_CACHE, debug_size=debug_size
    )

    X_test = df_test[feature_cols].values.astype(np.float32)
    segment_ids = df_test["segment_id"].values

    # Load Scaler
    mean_path = os.path.join(Config.WORKING_DIR, "scaler_mean.npy")
    scale_path = os.path.join(Config.WORKING_DIR, "scaler_scale.npy")

    if os.path.exists(mean_path) and os.path.exists(scale_path):
        mean = np.load(mean_path)
        scale = np.load(scale_path)
        # Manual standardization
        X_test = (X_test - mean) / scale
    else:
        print("Warning: Scaler files not found. Test data will not be scaled.")

    # Prediction Loop
    model.eval()
    predictions = []

    test_tensor = torch.tensor(X_test)
    dataset = TensorDataset(test_tensor)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            X_batch = batch[0].to(device)
            outputs = model(X_batch).squeeze()
            if outputs.ndim == 0:
                outputs = outputs.unsqueeze(0)
            predictions.extend(outputs.cpu().numpy())

    # Save Submission
    df_sub = pd.DataFrame({"segment_id": segment_ids, "time_to_eruption": predictions})

    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


# ---------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------
def run_pipeline():
    """
    Main entry point to execute the full training and submission pipeline.
    """
    Config.setup()
    device = torch.device(Config.DEVICE)

    # 1. Prepare Data
    train_loader, val_loader, input_dim, feature_cols = prepare_data(
        debug_size=Config.DEBUG_SAMPLE_SIZE
    )

    # 2. Train Model
    model = train_model(
        train_loader,
        val_loader,
        input_dim,
        device,
        epochs=Config.EPOCHS,
        lr=Config.LEARNING_RATE,
        patience=Config.EARLY_STOPPING_PATIENCE,
        model_save_path=Config.MODEL_SAVE_PATH,
    )

    # 3. Generate Submission
    generate_submission(
        model, feature_cols, device, debug_size=Config.DEBUG_SAMPLE_SIZE
    )
