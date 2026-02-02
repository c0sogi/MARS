import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import time

# Import from provided library
from library.data_loader import GnssWindowedDataset

# --- Model Definition ---


class TemporalConvNet(nn.Module):
    def __init__(self, input_channels=5, window_size=64, hidden_dim=128, output_dim=2):
        super(TemporalConvNet, self).__init__()

        self.window_size = window_size

        # Convolutional Block 1
        self.conv1 = nn.Conv1d(
            in_channels=input_channels, out_channels=32, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()

        # Convolutional Block 2
        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()

        # Convolutional Block 3
        self.conv3 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()

        # Fully Connected Layers
        # Flatten size: 128 channels * window_size
        self.flatten_dim = 128 * window_size

        self.fc1 = nn.Linear(self.flatten_dim, hidden_dim)
        self.relu_fc = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Input shape: (Batch, Window, Channels)
        # Conv1d expects: (Batch, Channels, Window)
        x = x.permute(0, 2, 1)

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        x = self.relu_fc(self.fc1(x))
        x = self.dropout(x)
        out = self.fc2(x)

        return out


# --- Training Logic ---


def train_model(model, train_loader, val_loader, config):
    """
    Trains the TemporalConvNet model.

    Args:
        model: PyTorch model instance.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        config: Dictionary containing hyperparameters (lr, epochs, patience, device).

    Returns:
        model: Trained model with best weights loaded.
        history: Dictionary of loss history.
    """
    device = config.get(
        "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    epochs = config.get("epochs", 10)
    patience = config.get("patience", 3)
    learning_rate = config.get("lr", 1e-3)

    model = model.to(device)

    # Loss and Optimizer
    criterion = nn.L1Loss()  # Mean Absolute Error is robust for GNSS
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, verbose=False
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    history = {"train_loss": [], "val_loss": []}

    print(f"Starting training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        start_time = time.time()

        # Training Phase
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            features = batch["features"].to(device)
            targets = batch["target"].to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * features.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)

        # Validation Phase
        model.eval()
        val_running_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                targets = batch["target"].to(device)

                outputs = model(features)
                loss = criterion(outputs, targets)

                val_running_loss += loss.item() * features.size(0)

        epoch_val_loss = val_running_loss / len(val_loader.dataset)

        # Update History
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)

        # Scheduler Step
        scheduler.step(epoch_val_loss)

        # Logging
        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{epochs} | Time: {elapsed:.1f}s | "
            f"Train Loss (MAE): {epoch_train_loss:.6f} | Val Loss (MAE): {epoch_val_loss:.6f}"
        )

        # Early Stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Load best weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


# --- Submission Logic ---


def generate_submission(
    model, test_metadata_path, input_dir, output_file, config, scaler
):
    """
    Generates predictions for the test set and saves to CSV.

    Args:
        model: Trained PyTorch model.
        test_metadata_path: Path to test_metadata.csv.
        input_dir: Root directory of input data.
        output_file: Path to save the submission CSV.
        config: Configuration dictionary.
        scaler: Fitted StandardScaler from training.
    """
    device = config.get(
        "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    batch_size = config.get("batch_size", 256)
    window_size = config.get("window_size", 64)

    # Load Metadata
    if not os.path.exists(test_metadata_path):
        raise FileNotFoundError(f"Test metadata not found at {test_metadata_path}")

    df_test_meta = pd.read_csv(test_metadata_path)

    # Create Dataset and Loader
    test_dataset = GnssWindowedDataset(
        metadata_df=df_test_meta,
        input_dir=input_dir,
        window_size=window_size,
        mode="test",
        scaler=scaler,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    model.eval()
    model.to(device)

    all_preds = []
    all_timestamps = []

    print("Generating predictions for test set...")

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            baseline = batch["baseline"].numpy()  # (Batch, 2) -> Lat, Lon
            timestamps = batch["timestamp"].numpy()

            # Predict Residuals (dLat, dLon)
            residuals = model(features).cpu().numpy()

            # Reconstruct: Prediction = Baseline + Residual
            # baseline is [Lat, Lon], residuals is [dLat, dLon]
            predictions = baseline + residuals

            all_preds.append(predictions)
            all_timestamps.append(timestamps)

    # Concatenate results
    if len(all_preds) == 0:
        print("No predictions generated.")
        return

    all_preds = np.concatenate(all_preds, axis=0)
    all_timestamps = np.concatenate(all_timestamps, axis=0)

    # The test_loader iterates in order of the metadata.
    # We can assign predictions back to the metadata dataframe directly if order is preserved.
    # GnssWindowedDataset preserves order of metadata rows.

    # Ensure lengths match
    if len(all_preds) != len(df_test_meta):
        print(
            f"Warning: Prediction count {len(all_preds)} != Metadata count {len(df_test_meta)}"
        )
        # In case of mismatch (e.g. missing files), we align by index or handle carefully.
        # But here we assume 1-to-1 mapping based on dataset implementation.

    df_test_meta["LatitudeDegrees"] = all_preds[:, 0]
    df_test_meta["LongitudeDegrees"] = all_preds[:, 1]

    # Prepare Submission DataFrame
    # Required columns: tripId, UnixTimeMillis, LatitudeDegrees, LongitudeDegrees
    submission_df = df_test_meta[
        ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    ]

    # Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    submission_df.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")
