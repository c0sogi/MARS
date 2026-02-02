import os
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import load_sensor_geometry, angular_dist_score, direction_to_angles
from library.model import PointNetBaseline, cosine_loss
from library.data import IceCubeBatchDataset


def train_epoch(model, batch_ids, meta_df, sensor_geo, optimizer, device, config):
    """
    Trains the model for one epoch over the specified list of batch files.
    """
    model.train()
    total_loss = 0.0
    num_steps = 0

    # Shuffle the order of batch files for this epoch
    # We copy to avoid modifying the original array if it's used elsewhere
    current_batch_ids = batch_ids.copy()
    np.random.shuffle(current_batch_ids)

    for batch_id in current_batch_ids:
        # Instantiate dataset for the current batch file
        # The dataset class handles caching logic internally
        try:
            dataset = IceCubeBatchDataset(
                batch_id=batch_id,
                meta_df=meta_df,
                sensor_geo=sensor_geo,
                mode="train",
                load_cached_data=True,
            )
        except Exception:
            # In case of file read errors or missing data, skip this batch
            continue

        if len(dataset) == 0:
            continue

        # Create DataLoader for mini-batching within the file
        loader = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=0,  # Avoid overhead for pre-loaded memory data
        )

        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            # Forward pass
            preds = model(X)

            # Compute loss
            loss = cosine_loss(preds, y)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_steps += 1

    return total_loss / max(1, num_steps)


def validate(model, batch_ids, meta_df, sensor_geo, device, config):
    """
    Evaluates the model on the validation set.
    Returns average loss and average Mean Angular Error (MAE).
    """
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    num_steps = 0

    with torch.no_grad():
        for batch_id in batch_ids:
            try:
                dataset = IceCubeBatchDataset(
                    batch_id=batch_id,
                    meta_df=meta_df,
                    sensor_geo=sensor_geo,
                    mode="train",  # mode is train because we need targets (y)
                    load_cached_data=True,
                )
            except Exception:
                continue

            if len(dataset) == 0:
                continue

            loader = DataLoader(
                dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
            )

            for X, y in loader:
                X = X.to(device)
                y = y.to(device)

                # Forward pass
                preds = model(X)

                # Compute Loss
                loss = cosine_loss(preds, y)
                total_loss += loss.item()

                # Compute MAE Metric
                # Normalize predictions to unit vectors
                pred_vecs = F.normalize(preds, p=2, dim=1)

                # Convert vectors back to angles
                az_pred, zen_pred = direction_to_angles(pred_vecs)

                # Prepare numpy arrays for metric calculation
                y_pred_np = torch.stack([az_pred, zen_pred], dim=1).cpu().numpy()
                y_true_np = y.cpu().numpy()

                # Calculate mean angular error for this batch
                mae = angular_dist_score(y_true_np, y_pred_np)
                total_mae += mae

                num_steps += 1

    avg_loss = total_loss / max(1, num_steps)
    avg_mae = total_mae / max(1, num_steps)

    return avg_loss, avg_mae


def run_training(config=Config):
    """
    Main function to orchestrate the training process.
    """
    # Initialize directories
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Set device
    device = torch.device(config.DEVICE)

    # Load Metadata
    train_meta = pd.read_parquet(config.TRAIN_META)
    val_meta = pd.read_parquet(config.VAL_META)

    # Handle Debug Mode
    if config.DEBUG:
        train_meta = train_meta.iloc[: config.DEBUG_SUBSET_SIZE]
        val_meta = val_meta.iloc[: config.DEBUG_SUBSET_SIZE]

    # Extract unique batch IDs
    train_batch_ids = train_meta["batch_id"].unique()
    val_batch_ids = val_meta["batch_id"].unique()

    # Load Sensor Geometry (shared across all datasets)
    sensor_geo = load_sensor_geometry(config.SENSOR_GEOMETRY_PATH)

    # Initialize Model
    model = PointNetBaseline(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        output_dim=config.OUTPUT_DIM,
        dropout=config.DROPOUT,
    ).to(device)

    # Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )

    # Training Loop Variables
    best_val_loss = float("inf")
    patience_counter = 0

    print(
        f"Starting training with {len(train_batch_ids)} train batches and {len(val_batch_ids)} val batches."
    )

    for epoch in range(config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_epoch(
            model, train_batch_ids, train_meta, sensor_geo, optimizer, device, config
        )

        # Validate
        val_loss, val_mae = validate(
            model, val_batch_ids, val_meta, sensor_geo, device, config
        )

        elapsed = time.time() - start_time

        # Print Metrics (Full Precision)
        print(
            f"Epoch {epoch+1} | Time: {elapsed:.2f}s | Train Loss: {train_loss} | Val Loss: {val_loss} | Val MAE: {val_mae}"
        )

        # Learning Rate Scheduling
        scheduler.step(val_loss)

        # Checkpointing and Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Loss: {best_val_loss}")
