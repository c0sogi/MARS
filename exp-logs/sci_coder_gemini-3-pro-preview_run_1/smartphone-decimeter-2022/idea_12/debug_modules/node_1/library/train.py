import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, Meters_to_WGS84
from library.data_preprocessing import prepare_training_data, prepare_test_data
from library.dataset import GNSSSequenceDataset, gnss_collate_fn
from library.model import AtrousResUNet
from library.loss import DeepSupervisionMAELoss


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        features = batch["features"].to(device)
        targets = batch["targets"].to(device)
        masks = batch["masks"].to(device)

        optimizer.zero_grad()

        # Forward pass
        # outputs is a list: [final_out, aux1, aux2]
        outputs = model(features)

        # Compute loss using DeepSupervisionMAELoss
        loss = criterion(outputs, targets, masks)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and MAE metrics for North and East components.
    """
    model.eval()
    running_loss = 0.0

    mae_north = 0.0
    mae_east = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            masks = batch["masks"].to(device)

            outputs = model(features)

            # Compute loss (Deep Supervision)
            loss = criterion(outputs, targets, masks)
            running_loss += loss.item()

            # Compute Metrics on Final Output only
            final_out = outputs[0]

            # Iterate through batch to handle variable lengths via mask
            for i in range(features.shape[0]):
                valid_len = int(masks[i].sum().item())
                if valid_len == 0:
                    continue

                # Extract valid sequence
                pred_seq = final_out[i, :, :valid_len].cpu().numpy()
                target_seq = targets[i, :, :valid_len].cpu().numpy()

                # Calculate absolute errors
                # Channel 0: North, Channel 1: East
                abs_diff = np.abs(pred_seq - target_seq)

                mae_north += abs_diff[0].sum()
                mae_east += abs_diff[1].sum()
                total_samples += valid_len

    avg_loss = running_loss / len(loader)
    avg_mae_north = mae_north / total_samples if total_samples > 0 else 0.0
    avg_mae_east = mae_east / total_samples if total_samples > 0 else 0.0

    return avg_loss, avg_mae_north, avg_mae_east


def generate_submission(model, device, scaler):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("\nPreparing test data for submission...")
    test_df = prepare_test_data(load_cached_data=True)

    test_dataset = GNSSSequenceDataset(test_df, mode="test", scaler=scaler)

    # Batch size 1 ensures simple reconstruction of sequences
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=gnss_collate_fn,
        num_workers=2,
    )

    model.eval()
    results = []

    print("Running inference on test set...")
    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            masks = batch["masks"]

            # Predict
            outputs = model(features)
            final_out = outputs[0].cpu().numpy()  # (1, 2, L)

            # Metadata for reconstruction
            trip_id = batch["trip_ids"][0]
            wls_pos = batch["wls_pos"][0]  # (L, 2) [lat, lon]
            timestamps = batch["timestamps"][0]

            valid_len = int(masks[0].sum().item())

            # Extract valid predictions: (2, L) -> (L, 2)
            # Channel 0 is North (dLat), Channel 1 is East (dLon)
            preds = final_out[0, :, :valid_len].T

            delta_north = preds[:, 0]
            delta_east = preds[:, 1]

            base_lat = wls_pos[:valid_len, 0]
            base_lon = wls_pos[:valid_len, 1]
            valid_timestamps = timestamps[:valid_len]

            # Convert meters back to WGS84 coordinates
            pred_lat, pred_lon = Meters_to_WGS84(
                base_lat, base_lon, delta_north, delta_east
            )

            # Create DataFrame for this trip
            trip_df = pd.DataFrame(
                {
                    "tripId": [trip_id] * valid_len,
                    "UnixTimeMillis": valid_timestamps,
                    "LatitudeDegrees": pred_lat,
                    "LongitudeDegrees": pred_lon,
                }
            )
            results.append(trip_df)

    # Concatenate all trips
    submission_df = pd.concat(results, ignore_index=True)

    # Load sample submission to ensure correct format and rows
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

    # Create merge keys
    submission_df["key"] = (
        submission_df["tripId"] + "_" + submission_df["UnixTimeMillis"].astype(str)
    )
    sample_sub["key"] = (
        sample_sub["tripId"] + "_" + sample_sub["UnixTimeMillis"].astype(str)
    )

    # Merge predictions
    final_sub = sample_sub.drop(columns=["LatitudeDegrees", "LongitudeDegrees"]).merge(
        submission_df[["key", "LatitudeDegrees", "LongitudeDegrees"]],
        on="key",
        how="left",
    )

    # Handle missing predictions (if any) via interpolation
    final_sub["LatitudeDegrees"] = (
        final_sub["LatitudeDegrees"]
        .interpolate(method="linear")
        .fillna(method="bfill")
        .fillna(method="ffill")
    )
    final_sub["LongitudeDegrees"] = (
        final_sub["LongitudeDegrees"]
        .interpolate(method="linear")
        .fillna(method="bfill")
        .fillna(method="ffill")
    )

    # Cleanup
    final_sub = final_sub.drop(columns=["key"])

    # Save
    save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    final_sub.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")


def run_training(epochs=Config.NUM_EPOCHS, debug_size=None):
    """
    Main function to run the training pipeline.

    Args:
        epochs (int): Number of training epochs.
        debug_size (int, optional): Number of drives to load for debugging.
    """
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training pipeline on device: {device}")

    # Override Config debug size if provided
    if debug_size is not None:
        Config.DEBUG_SAMPLE_SIZE = debug_size
        print(f"Debug mode enabled. Using {debug_size} drives.")

    # 1. Prepare Data
    # Note: prepare_training_data uses Config.DEBUG_SAMPLE_SIZE internally if set
    train_df, val_df = prepare_training_data(load_cached_data=True)

    train_dataset = GNSSSequenceDataset(train_df, mode="train")
    val_dataset = GNSSSequenceDataset(val_df, mode="train", scaler=train_dataset.scaler)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=gnss_collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=gnss_collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # 2. Initialize Model & Training Components
    model = AtrousResUNet(
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.OUT_CHANNELS,
        base_dim=Config.HIDDEN_DIM,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
    )

    criterion = DeepSupervisionMAELoss(weights=Config.LOSS_WEIGHTS).to(device)

    # 3. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"\nTraining for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae_n, val_mae_e = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val MAE North: {val_mae_n:.6f}m | "
            f"Val MAE East: {val_mae_e:.6f}m"
        )

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved (Loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # 4. Generate Submission
    print("\nLoading best model for submission...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    generate_submission(model, device, train_dataset.scaler)
