import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from library.config import Config
from library.model import SensorFusionTCN
from library.data_loader import SmartphoneDataset


def train_model(
    train_meta_path=Config.TRAIN_METADATA_PATH,
    val_meta_path=Config.VAL_METADATA_PATH,
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
):
    """
    Trains the SensorFusionTCN model using the provided metadata.
    """
    print(f"Initializing Training on {Config.DEVICE}...")

    # 1. Load Datasets
    print("Loading Training Dataset...")
    train_dataset = SmartphoneDataset(train_meta_path, Config.WINDOW_SIZE, mode="train")
    print("Loading Validation Dataset...")
    val_dataset = SmartphoneDataset(val_meta_path, Config.WINDOW_SIZE, mode="val")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # 2. Initialize Model
    model = SensorFusionTCN(
        num_inputs=Config.NUM_FEATURES,
        num_channels=[Config.HIDDEN_CHANNELS] * Config.NUM_LAYERS,
        kernel_size=Config.KERNEL_SIZE,
        dropout=Config.DROPOUT,
    ).to(Config.DEVICE)

    # 3. Setup Optimization
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 4. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    weights_path = os.path.join(Config.WORKING_DIR, "model_weights.pth")

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss_sum = 0.0
        train_samples = 0

        for features, targets in train_loader:
            features = features.to(Config.DEVICE)
            targets = targets.to(Config.DEVICE)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * features.size(0)
            train_samples += features.size(0)

        avg_train_loss = train_loss_sum / train_samples if train_samples > 0 else 0.0

        # --- Validation ---
        model.eval()
        val_loss_sum = 0.0
        val_samples = 0

        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(Config.DEVICE)
                targets = targets.to(Config.DEVICE)

                outputs = model(features)
                loss = criterion(outputs, targets)

                val_loss_sum += loss.item() * features.size(0)
                val_samples += features.size(0)

        avg_val_loss = val_loss_sum / val_samples if val_samples > 0 else 0.0

        print(
            f"Epoch {epoch+1}/{epochs} | Train MAE: {avg_train_loss:.10f} | Val MAE: {avg_val_loss:.10f}"
        )

        # --- Early Stopping & Checkpointing ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), weights_path)
            # print(f"  Best model saved to {weights_path}")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    return model


def generate_submission(test_meta_path=Config.TEST_METADATA_PATH, model=None):
    """
    Generates the submission file using the trained model.
    """
    print("Generating Submission...")

    # 1. Load Test Dataset
    test_dataset = SmartphoneDataset(test_meta_path, Config.WINDOW_SIZE, mode="test")
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    # 2. Load Model
    if model is None:
        model = SensorFusionTCN(
            num_inputs=Config.NUM_FEATURES,
            num_channels=[Config.HIDDEN_CHANNELS] * Config.NUM_LAYERS,
            kernel_size=Config.KERNEL_SIZE,
            dropout=Config.DROPOUT,
        ).to(Config.DEVICE)

        weights_path = os.path.join(Config.WORKING_DIR, "model_weights.pth")
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=Config.DEVICE))
            print(f"Loaded weights from {weights_path}")
        else:
            print("Warning: No model weights found. Using untrained model.")

    model.eval()

    # 3. Inference
    all_preds = []
    all_baselines = []

    with torch.no_grad():
        for features, wls_baseline in test_loader:
            features = features.to(Config.DEVICE)

            # Predict residuals (Delta Lat, Delta Lon)
            residuals = model(features).cpu().numpy()

            all_preds.append(residuals)
            all_baselines.append(wls_baseline.numpy())

    if not all_preds:
        print("No predictions generated.")
        return

    residuals = np.concatenate(all_preds, axis=0)
    baselines = np.concatenate(all_baselines, axis=0)

    # 4. Reconstruct Absolute Coordinates
    # Prediction = Baseline + Residual
    final_lat = baselines[:, 0] + residuals[:, 0]
    final_lon = baselines[:, 1] + residuals[:, 1]

    # 5. Construct Submission DataFrame
    # Retrieve metadata corresponding to the valid windows
    # SmartphoneDataset filters rows; we need to get the corresponding rows from the full_df
    valid_indices = test_dataset.indices
    submission_df = test_dataset.full_df.iloc[valid_indices].copy()

    submission_df["LatitudeDegrees"] = final_lat
    submission_df["LongitudeDegrees"] = final_lon

    # Format according to sample submission
    output_cols = ["tripId", "utcTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    final_sub = submission_df[output_cols].rename(
        columns={"utcTimeMillis": "UnixTimeMillis"}
    )

    output_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    final_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} with {len(final_sub)} rows.")
