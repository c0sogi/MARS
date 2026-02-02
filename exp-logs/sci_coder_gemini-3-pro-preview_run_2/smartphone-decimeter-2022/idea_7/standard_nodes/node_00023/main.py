import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import random

# Import from provided library files
from library.config import Config
from library.dataset import get_dataset
from library.model import LocalAttentionTransformer
from library.engine import train_one_epoch, generate_submission
from library.utils import (
    meters_to_deg,
    calculate_competition_metric,
    haversine_distance,
)


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_and_analyze(model, val_loader, device):
    """
    Runs validation inference, computes the competition metric,
    and performs failure analysis by correlating errors with input features.
    """
    model.eval()

    pred_records = []
    gt_records = []

    # Lists to store error and feature values for analysis
    errors_list = []
    feature_data = {"mean_cn0": [], "mean_unc": [], "sat_count": []}

    # Feature indices based on Config.INPUT_FEATURES:
    # ['rel_lat_m', 'rel_lon_m', 'vel_lat_m', 'vel_lon_m', 'vel_alt_m', 'mean_cn0', 'mean_unc', 'sat_count']
    feat_idx_map = {"mean_cn0": 5, "mean_unc": 6, "sat_count": 7}

    print("Running validation and failure analysis...")

    with torch.no_grad():
        for batch_x, batch_y, meta in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Forward pass
            outputs = model(batch_x)

            # Move to CPU
            preds_m = outputs.cpu().numpy()
            targets_m = batch_y.cpu().numpy()
            inputs_np = batch_x.cpu().numpy()

            # Metadata
            trip_ids = meta["tripId"]
            timestamps = meta["UnixTimeMillis"].numpy()
            wls_lats = meta["wls_lat"].numpy()
            wls_lons = meta["wls_lon"].numpy()

            # Convert predictions to degrees
            d_lat_pred, d_lon_pred = meters_to_deg(
                preds_m[:, 0], preds_m[:, 1], wls_lats
            )
            pred_lats = wls_lats + d_lat_pred
            pred_lons = wls_lons + d_lon_pred

            # Convert targets to degrees (to reconstruct GT)
            d_lat_gt, d_lon_gt = meters_to_deg(
                targets_m[:, 0], targets_m[:, 1], wls_lats
            )
            gt_lats = wls_lats + d_lat_gt
            gt_lons = wls_lons + d_lon_deg

            # Calculate Haversine error for this batch
            batch_errors = haversine_distance(pred_lats, pred_lons, gt_lats, gt_lons)
            errors_list.extend(batch_errors)

            # Extract features from the center of the window (index = window_size // 2)
            center_idx = Config.WINDOW_SIZE // 2

            # Note: inputs are normalized, but correlation is scale-invariant, so it's fine.
            for feat_name, idx in feat_idx_map.items():
                # Shape: (Batch, Window, Features) -> (Batch, Features)
                feat_vals = inputs_np[:, center_idx, idx]
                feature_data[feat_name].extend(feat_vals)

            # Store records for metric calculation
            for i in range(len(trip_ids)):
                pred_records.append(
                    {
                        "tripId": trip_ids[i],
                        "UnixTimeMillis": timestamps[i],
                        "LatitudeDegrees": pred_lats[i],
                        "LongitudeDegrees": pred_lons[i],
                    }
                )
                gt_records.append(
                    {
                        "tripId": trip_ids[i],
                        "UnixTimeMillis": timestamps[i],
                        "LatitudeDegrees": gt_lats[i],
                        "LongitudeDegrees": gt_lons[i],
                    }
                )

    # Create DataFrames
    df_pred = pd.DataFrame(pred_records)
    df_gt = pd.DataFrame(gt_records)

    # 1. Compute Competition Metric
    final_metric = calculate_competition_metric(df_pred, df_gt)
    print(f"Final Validation Metric: {final_metric}")

    # 2. Failure Analysis
    print("\nFailure Analysis (Correlation with Error Magnitude):")
    analysis_df = pd.DataFrame(
        {
            "Error": errors_list,
            "mean_cn0": feature_data["mean_cn0"],
            "mean_unc": feature_data["mean_unc"],
            "sat_count": feature_data["sat_count"],
        }
    )

    correlations = analysis_df.corr()["Error"].drop("Error")
    print(correlations)

    return final_metric


def main():
    # 1. Configuration Override for Fast Baseline
    Config.EPOCHS = 5  # Reduced from 30 for speed
    Config.BATCH_SIZE = 512  # Increased for A100 efficiency
    Config.PATIENCE = 3

    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Initializing Datasets...")
    # load_cached_data=True will use parquet files in ./working/idea_7/ if they exist
    train_dataset = get_dataset("train", load_cached_data=True)
    val_dataset = get_dataset("val", load_cached_data=True)

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

    # 3. Model Initialization
    model = LocalAttentionTransformer().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 4. Training Loop
    print(f"Starting training for {Config.EPOCHS} epochs...")
    best_val_loss = float("inf")

    # Scheduler for training loop
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, verbose=True
    )

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validation (Loss only for speed during loop)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y, _ in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
        val_loss /= len(val_dataset)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print("  Saved best model.")

    # 5. Load Best Model
    print("Loading best model for analysis...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    # 6. Validation Assessment & Failure Analysis
    val_metric = validate_and_analyze(model, val_loader, device)

    # 7. Submission Generation
    threshold = 4.256982128481356
    if val_metric < threshold:
        print(
            f"Validation metric {val_metric} is below threshold {threshold}. Generating submission..."
        )

        # Load Test Data
        test_dataset = get_dataset("test", load_cached_data=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # Generate Submission
        generate_submission(model, test_loader, device, Config.SUBMISSION_FILE)
    else:
        print(
            f"Validation metric {val_metric} did not meet threshold {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    main()
