import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, enu_to_geodetic
from library.data_processing import process_dataset
from library.dataset import GNSSSequenceDataset, collate_padded
from library.model import UNet1D


def train_model(load_cached_data=True):
    """
    Trains the 1D U-Net model and saves the best weights.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from cache.

    Returns:
        tuple: (trained_model, feature_stats)
    """
    set_seed(Config.SEED)

    # 1. Load and Process Data
    print("Loading training data...")
    train_df = process_dataset(
        Config.TRAIN_METADATA_PATH,
        load_cached_data=load_cached_data,
        split_name="train",
    )
    print("Loading validation data...")
    val_df = process_dataset(
        Config.VAL_METADATA_PATH, load_cached_data=load_cached_data, split_name="val"
    )

    # 2. Create Datasets
    # Train dataset computes normalization stats
    print("Creating training dataset...")
    train_dataset = GNSSSequenceDataset(
        train_df, split_name="train", load_cached_data=load_cached_data
    )
    feature_stats = train_dataset.stats

    # Validation dataset uses training stats for consistent normalization
    print("Creating validation dataset...")
    val_dataset = GNSSSequenceDataset(
        val_df,
        split_name="val",
        feature_stats=feature_stats,
        load_cached_data=load_cached_data,
    )

    # 3. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_padded,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_padded,
        pin_memory=True,
    )

    # 4. Initialize Model
    device = Config.DEVICE
    model = UNet1D(
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.OUTPUT_CHANNELS,
        base_filters=Config.BASE_FILTERS,
        depth=Config.DEPTH,
        kernel_size=Config.KERNEL_SIZE,
        dropout=Config.DROPOUT,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )
    criterion = nn.L1Loss(reduction="none")

    # 5. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    model_save_path = os.path.join(Config.WORKING_DIR, "model_weights.pth")

    # Remove existing weights to avoid loading incompatible state if training fails to improve
    if os.path.exists(model_save_path):
        os.remove(model_save_path)

    print(f"Starting training on {device} for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss_sum = 0.0
        train_samples = 0

        for batch in train_loader:
            features = batch["features"].to(device)  # (B, C, L)
            targets = batch["targets"].to(device)  # (B, C, L)
            lengths = batch["lengths"].to(device)  # (B,)

            optimizer.zero_grad()

            outputs = model(features)

            # Create mask for variable lengths to ignore padded regions in loss
            # outputs shape: (B, C, L)
            max_len = outputs.size(2)
            # Create mask: (B, L) -> True where valid
            mask = torch.arange(max_len, device=device).unsqueeze(
                0
            ) < lengths.unsqueeze(1)
            # Expand to (B, C, L)
            mask = mask.unsqueeze(1).expand_as(outputs)

            loss_raw = criterion(outputs, targets)
            # Apply mask and compute mean over valid elements
            loss = (loss_raw * mask.float()).sum() / (mask.float().sum() + 1e-8)

            loss.backward()
            optimizer.step()

            # Weighted sum for accurate epoch average
            train_loss_sum += loss.item() * len(batch["drive_id"])
            train_samples += len(batch["drive_id"])

        avg_train_loss = train_loss_sum / train_samples

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                targets = batch["targets"].to(device)
                lengths = batch["lengths"].to(device)

                outputs = model(features)

                max_len = outputs.size(2)
                mask = torch.arange(max_len, device=device).unsqueeze(
                    0
                ) < lengths.unsqueeze(1)
                mask = mask.unsqueeze(1).expand_as(outputs)

                loss_raw = criterion(outputs, targets)
                loss = (loss_raw * mask.float()).sum() / (mask.float().sum() + 1e-8)

                val_loss_sum += loss.item() * len(batch["drive_id"])
                val_samples += len(batch["drive_id"])

        avg_val_loss = val_loss_sum / val_samples

        scheduler.step(avg_val_loss)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} - Train Loss: {avg_train_loss:.9f} - Val Loss: {avg_val_loss:.9f}"
        )

        # Early Stopping and Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"  New best model saved to {model_save_path}")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(
                    f"Early stopping triggered after {patience_counter} epochs without improvement."
                )
                break

    # Load best model for return
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    return model, feature_stats


def generate_submission(model, feature_stats, load_cached_data=True):
    """
    Generates predictions for the test set using the trained model and saves the submission file.

    Args:
        model (nn.Module): Trained PyTorch model.
        feature_stats (dict): Normalization statistics from training.
        load_cached_data (bool): Whether to load pre-processed test data from cache.
    """
    set_seed(Config.SEED)
    device = Config.DEVICE

    print("Loading test data...")
    test_df = process_dataset(
        Config.TEST_METADATA_PATH, load_cached_data=load_cached_data, split_name="test"
    )

    print("Creating test dataset...")
    test_dataset = GNSSSequenceDataset(
        test_df,
        split_name="test",
        feature_stats=feature_stats,
        load_cached_data=load_cached_data,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_padded,
        pin_memory=True,
    )

    model.eval()
    results = []

    print("Running inference...")
    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            lengths = batch["lengths"]  # CPU
            timestamps = batch["timestamps"]  # CPU
            drive_ids = batch["drive_id"]
            phone_names = batch["phone_name"]

            outputs = model(features).cpu().numpy()  # (B, C, L)

            # Iterate through batch
            for i in range(len(drive_ids)):
                length = lengths[i]
                drive_id = drive_ids[i]
                phone_name = phone_names[i]

                # Extract valid sequence parts (remove padding)
                # outputs[i] is (C, L_padded) -> Transpose to (L_padded, C)
                pred_seq = outputs[i].T[:length]  # (L, 2)
                time_seq = timestamps[i][:length].numpy()

                for t, (lat_res, lon_res) in enumerate(pred_seq):
                    results.append(
                        {
                            "drive_id": drive_id,
                            "phone_name": phone_name,
                            "UnixTimeMillis": time_seq[t],
                            "lat_res_m": lat_res,
                            "lon_res_m": lon_res,
                        }
                    )

    # Create DataFrame from predictions
    pred_df = pd.DataFrame(results)

    # Merge with original test_df to get Baseline coordinates
    # Ensure types match for merge
    pred_df["UnixTimeMillis"] = pred_df["UnixTimeMillis"].astype(np.int64)
    test_df["UnixTimeMillis"] = test_df["UnixTimeMillis"].astype(np.int64)

    # Inner join matches predictions to baselines
    merged_df = pd.merge(
        test_df, pred_df, on=["drive_id", "phone_name", "UnixTimeMillis"], how="inner"
    )

    # Apply corrections: Baseline + Predicted Residuals (ENU -> Geodetic)
    pred_lats, pred_lons = enu_to_geodetic(
        merged_df["lat_res_m"].values,
        merged_df["lon_res_m"].values,
        merged_df["BaselineLat"].values,
        merged_df["BaselineLon"].values,
    )

    merged_df["LatitudeDegrees"] = pred_lats
    merged_df["LongitudeDegrees"] = pred_lons

    # Format submission
    # Ensure all required columns are present. tripId is in test_df from metadata.
    submission_df = merged_df[
        ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    ]

    output_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
