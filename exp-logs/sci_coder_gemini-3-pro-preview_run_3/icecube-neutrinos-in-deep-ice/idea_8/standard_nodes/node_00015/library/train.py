import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from library.config import Config
from library.data import IceCubeDataset, process_batch
from library.model import DV_AGN
from library.loss import CosineSimilarityLoss
from library.utils import seed_everything, angular_error
from library.geometry import load_sensor_geometry


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    """
    Trains the model for one epoch over a specific DataLoader (representing one batch file).
    """
    model.train()
    running_loss = 0.0
    count = 0

    for X_raw, X_canon, targets in loader:
        X_raw = X_raw.to(device)
        X_canon = X_canon.to(device)
        azimuth = targets[:, 0].to(device)
        zenith = targets[:, 1].to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = model(X_raw, X_canon)

        # Loss calculation
        loss = criterion(preds, azimuth, zenith)

        # Backward pass
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * X_raw.size(0)
        count += X_raw.size(0)

    return running_loss, count


def validate(model, loader, device):
    """
    Evaluates the model on a specific DataLoader.
    Returns the sum of angular errors and the count of samples.
    """
    model.eval()
    running_error = 0.0
    count = 0

    with torch.no_grad():
        for X_raw, X_canon, targets in loader:
            X_raw = X_raw.to(device)
            X_canon = X_canon.to(device)
            azimuth = targets[:, 0].numpy()
            zenith = targets[:, 1].numpy()

            preds = model(X_raw, X_canon)

            # Calculate angular error (metric)
            errors = angular_error(preds, azimuth, zenith)
            running_error += np.sum(errors)
            count += len(errors)

    return running_error, count


def train_model(
    load_cached_data=True, epochs=Config.EPOCHS, patience=3, debug=False, save_path=None
):
    """
    Main function to train the DV-AGN model.

    Args:
        load_cached_data (bool): Whether to load pre-processed .npy files from cache.
        epochs (int): Number of training epochs.
        patience (int): Early stopping patience.
        debug (bool): If True, trains on a small subset of data for debugging.
        save_path (str): Path to save the best model checkpoint. Defaults to Config.WORKING_DIR/model.pth.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    Config.setup()
    device = Config.DEVICE

    if save_path is None:
        save_path = os.path.join(Config.WORKING_DIR, "model.pth")

    print(f"Device: {device}")
    print("Loading sensor geometry...")
    sensor_map = load_sensor_geometry(Config.SENSOR_GEO_PATH)

    # 2. Load Metadata
    print("Loading metadata...")
    train_meta = pd.read_parquet(
        os.path.join(Config.METADATA_DIR, "train_metadata.parquet")
    )
    val_meta = pd.read_parquet(
        os.path.join(Config.METADATA_DIR, "val_metadata.parquet")
    )

    if debug:
        print("DEBUG MODE: Limiting training data.")
        train_meta = train_meta.iloc[:10000]
        val_meta = val_meta.iloc[:2000]

    # Get unique batch IDs
    train_batches = train_meta["batch_id"].unique()
    val_batches = val_meta["batch_id"].unique()

    # 3. Model Initialization
    print("Initializing model...")
    model = DV_AGN().to(device)
    criterion = CosineSimilarityLoss()

    # Optimizer
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Calculate total steps for Scheduler
    # We need to estimate the number of steps per epoch.
    # Since we load data batch-by-batch, we sum the number of batches in each file.
    # For exactness, we can calculate:
    total_samples = len(train_meta)
    steps_per_epoch = int(np.ceil(total_samples / Config.BATCH_SIZE))
    total_steps = steps_per_epoch * epochs

    scheduler = OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        total_steps=total_steps,
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1000,
    )

    # 4. Training Loop
    best_val_mae = float("inf")
    no_improve_epochs = 0

    print("Starting training...")

    for epoch in range(epochs):
        # Training Phase
        model.train()
        total_train_loss = 0.0
        total_train_samples = 0

        # Shuffle batch order for randomness
        np.random.shuffle(train_batches)

        for batch_id in train_batches:
            # Load and process batch
            X_raw, X_canon, targets = process_batch(
                batch_id,
                train_meta,
                sensor_map,
                mode="train",
                load_cached_data=load_cached_data,
            )

            # Create DataLoader
            dataset = IceCubeDataset(X_raw, X_canon, targets)
            loader = DataLoader(
                dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=True,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            # Train on this batch file
            batch_loss, batch_count = train_one_epoch(
                model, loader, optimizer, scheduler, criterion, device
            )

            total_train_loss += batch_loss
            total_train_samples += batch_count

            # Cleanup to save memory
            del X_raw, X_canon, targets, dataset, loader
            gc.collect()

        avg_train_loss = (
            total_train_loss / total_train_samples if total_train_samples > 0 else 0.0
        )

        # Validation Phase
        model.eval()
        total_val_error = 0.0
        total_val_samples = 0

        for batch_id in val_batches:
            X_raw, X_canon, targets = process_batch(
                batch_id,
                val_meta,
                sensor_map,
                mode="val",
                load_cached_data=load_cached_data,
            )

            dataset = IceCubeDataset(X_raw, X_canon, targets)
            loader = DataLoader(
                dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            batch_error, batch_count = validate(model, loader, device)

            total_val_error += batch_error
            total_val_samples += batch_count

            del X_raw, X_canon, targets, dataset, loader
            gc.collect()

        avg_val_mae = (
            total_val_error / total_val_samples if total_val_samples > 0 else 0.0
        )

        # Logging (Full Precision)
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {avg_train_loss}")
        print(f"Val MAE: {avg_val_mae}")

        # Checkpointing & Early Stopping
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            print(f"Validation MAE improved. Saving model to {save_path}")
            torch.save(model.state_dict(), save_path)
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f"No improvement. Patience: {no_improve_epochs}/{patience}")

        if no_improve_epochs >= patience:
            print("Early stopping triggered.")
            break

    print("Training complete.")
