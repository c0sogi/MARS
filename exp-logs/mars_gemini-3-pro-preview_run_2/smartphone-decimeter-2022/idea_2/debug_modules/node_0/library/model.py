import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from library.config import Config
from library.utils import local_meters_to_wgs84


class Attention(nn.Module):
    """
    Simple Attention Mechanism to aggregate LSTM hidden states.
    Computes a weighted sum of the sequence outputs.
    """

    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)
        # weights shape: (batch_size, seq_len, 1)
        weights = self.attention(x)
        weights = torch.softmax(weights, dim=1)

        # Context vector shape: (batch_size, hidden_dim)
        context = torch.sum(x * weights, dim=1)
        return context


class BiLSTMRegressor(nn.Module):
    """
    Sequence-Aware Residual Regressor using a Bidirectional LSTM backbone.
    Predicts the (DeltaEast, DeltaNorth) correction for the center timestamp of the window.
    """

    def __init__(
        self,
        input_dim=len(Config.INPUT_FEATURES),
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
        output_dim=len(Config.TARGET_COLUMNS),
    ):
        super(BiLSTMRegressor, self).__init__()

        # Input projection to hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()

        # Bidirectional LSTM Backbone
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Attention Layer (Input dim is hidden_dim * 2 because of bidirectionality)
        self.attention = Attention(hidden_dim * 2)

        # Output Regression Head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)

        # Project features
        x = self.input_proj(x)
        x = self.relu(x)

        # Pass through LSTM
        # lstm_out shape: (batch_size, seq_len, 2 * hidden_dim)
        lstm_out, _ = self.lstm(x)

        # Aggregate sequence using attention
        # context shape: (batch_size, 2 * hidden_dim)
        context = self.attention(lstm_out)

        # Predict residuals
        out = self.fc(context)
        return out


def train_model(model, train_loader, val_loader):
    """
    Trains the BiLSTMRegressor model with Early Stopping and LR Scheduling.

    Args:
        model: The PyTorch model instance.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.

    Returns:
        The trained model with the best validation weights loaded.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Use L1 Loss (MAE) which is robust to outliers
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.CACHE_DIR, "bilstm_model.pth")

    print(f"Starting training on {device}...")

    for epoch in range(Config.EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item() * x.size(0)

        val_loss /= len(val_loader.dataset)

        # Update Scheduler
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train MAE: {train_loss:.6f} | Val MAE: {val_loss:.6f}"
        )

        # --- Early Stopping & Checkpointing ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"Training complete. Best Val MAE: {best_val_loss:.6f}")

    # Load best model state
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    return model


def generate_submission(model, test_loader):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        model: The trained PyTorch model.
        test_loader: DataLoader for test data (must be sequential, shuffle=False).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions = []

    print("Generating predictions...")
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            outputs = model(x)
            predictions.append(outputs.cpu().numpy())

    # Concatenate all predictions into a single array
    # Shape: (N_test, 2) -> [DeltaEast, DeltaNorth]
    pred_deltas = np.concatenate(predictions, axis=0)

    # Retrieve Metadata and Baseline WLS from the dataset
    # The dataset df is aligned with the loader because shuffle=False for test
    test_df = test_loader.dataset.df

    wls_lat = test_df["WlsLat"].values
    wls_lon = test_df["WlsLon"].values

    # Reconstruct absolute coordinates from predicted residuals
    pred_lat, pred_lon = local_meters_to_wgs84(
        wls_lat,
        wls_lon,
        pred_deltas[:, 0],  # DeltaEast
        pred_deltas[:, 1],  # DeltaNorth
    )

    # Create submission dataframe
    submission_df = pd.DataFrame(
        {
            "tripId": test_df["tripId"],
            "UnixTimeMillis": test_df["UnixTimeMillis"],
            "LatitudeDegrees": pred_lat,
            "LongitudeDegrees": pred_lon,
        }
    )

    # Save submission
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
