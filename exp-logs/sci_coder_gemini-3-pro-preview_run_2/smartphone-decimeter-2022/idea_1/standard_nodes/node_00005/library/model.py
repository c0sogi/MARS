import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from library.config import Config


class WindowedMLP(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) that processes a flattened window of sensor features
    to predict latitude and longitude residuals.
    """

    def __init__(
        self,
        input_dim=Config.INPUT_DIM,
        hidden_layers=Config.HIDDEN_LAYERS,
        output_dim=Config.OUTPUT_DIM,
        dropout_rate=Config.DROPOUT_RATE,
    ):
        super(WindowedMLP, self).__init__()

        layers = []
        curr_dim = input_dim

        # Construct hidden layers dynamically based on configuration
        for h_dim in hidden_layers:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            curr_dim = h_dim

        # Output layer (Predicts Lat/Lon residuals)
        layers.append(nn.Linear(curr_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=Config.EPOCHS,
    learning_rate=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    patience=Config.EARLY_STOPPING_PATIENCE,
    device=Config.DEVICE,
    checkpoint_path=Config.MODEL_CHECKPOINT_PATH,
):
    """
    Trains the WindowedMLP model with Early Stopping and Learning Rate Scheduling.
    """

    model = model.to(device)

    # Loss function: Mean Absolute Error is robust to outliers in GNSS data
    criterion = nn.L1Loss()

    # Optimizer: AdamW for decoupled weight decay
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Scheduler: Reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)

        # Update Scheduler
        scheduler.step(val_loss)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs}: Train Loss = {train_loss}, Val Loss = {val_loss}"
        )

        # --- Early Stopping & Checkpointing ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Validation loss improved. Model saved to {checkpoint_path}")
        else:
            patience_counter += 1
            print(
                f"  Validation loss did not improve. Patience: {patience_counter}/{patience}"
            )

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Load the best model weights before returning
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Loaded best model from checkpoint.")

    return model


def generate_submission(
    model,
    test_loader,
    meta_list,
    df_test_original,
    submission_path=Config.SUBMISSION_FILE_PATH,
    device=Config.DEVICE,
):
    """
    Generates predictions for the test set, reconstructs coordinates, and saves the submission file.

    Args:
        model: Trained PyTorch model.
        test_loader: DataLoader for test data.
        meta_list: List of (tripId, timestamp) corresponding to test_loader samples.
        df_test_original: DataFrame containing original WLS positions for reconstruction.
        submission_path: Path to save the CSV.
        device: Computation device.
    """

    model = model.to(device)
    model.eval()

    predictions = []

    print("Generating predictions...")
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)

    # Create DataFrame from predictions and metadata
    # meta_list aligns 1:1 with the predictions array
    trip_ids = [m[0] for m in meta_list]
    timestamps = [m[1] for m in meta_list]

    df_pred = pd.DataFrame(
        {
            Config.COL_TRIP_ID: trip_ids,
            Config.COL_UNIX_TIME: timestamps,
            "LatResidual": predictions[:, 0],
            "LonResidual": predictions[:, 1],
        }
    )

    # Merge with original WLS positions to calculate final coordinates
    # df_test_original must contain [tripId, UnixTimeMillis, WlsLat, WlsLon]
    df_merged = pd.merge(
        df_pred,
        df_test_original[
            [
                Config.COL_TRIP_ID,
                Config.COL_UNIX_TIME,
                Config.FEAT_WLS_LAT,
                Config.FEAT_WLS_LON,
            ]
        ],
        on=[Config.COL_TRIP_ID, Config.COL_UNIX_TIME],
        how="left",
    )

    # Apply residuals: Pred = Baseline + Residual / ScaleFactor
    # Unscale the predictions (Cite solution_lesson_node_00001)
    df_merged[Config.COL_LATITUDE] = df_merged[Config.FEAT_WLS_LAT] + (
        df_merged["LatResidual"] / Config.TARGET_SCALE_FACTOR
    )
    df_merged[Config.COL_LONGITUDE] = df_merged[Config.FEAT_WLS_LON] + (
        df_merged["LonResidual"] / Config.TARGET_SCALE_FACTOR
    )

    # Load the submission template to ensure we output exactly the required rows in order
    # Prefer generated test_metadata.csv as it is cleaner, fallback to sample_submission.csv
    if os.path.exists(Config.TEST_METADATA_PATH):
        df_template = pd.read_csv(Config.TEST_METADATA_PATH)
    else:
        df_template = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

    # Filter template to keys
    required_cols = [Config.COL_TRIP_ID, Config.COL_UNIX_TIME]
    df_template = df_template[required_cols]

    # Merge predictions into the template
    df_final = pd.merge(df_template, df_merged, on=required_cols, how="left")

    # Select final columns for submission
    submission_cols = [
        Config.COL_TRIP_ID,
        Config.COL_UNIX_TIME,
        Config.COL_LATITUDE,
        Config.COL_LONGITUDE,
    ]
    df_final = df_final[submission_cols]

    # Save to CSV
    df_final.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    return df_final
