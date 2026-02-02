import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import get_dataloaders
from library.model import HybridResUNetGRU
from library.utils import cartesian_to_wgs84, haversine_distance


def set_seed(seed):
    """Sets random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Runs one training epoch."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        # features: (B, C, T), targets: (B, 2, T), mask: (B, T)
        features = batch["features"].to(device)
        targets = batch["targets"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()

        outputs = model(features)  # (B, 2, T)

        # Apply mask to select only valid ground truth points
        # Permute outputs/targets to (B, T, 2) for boolean indexing if needed,
        # or just index directly since mask is (B, T)

        # We need to calculate loss only on valid time steps
        # Mask shape is (B, T). Expand to (B, 2, T) to mask both coordinates
        mask_expanded = mask.unsqueeze(1).expand_as(targets)

        if mask_expanded.sum() == 0:
            continue

        valid_outputs = outputs[mask_expanded]
        valid_targets = targets[mask_expanded]

        loss = criterion(valid_outputs, valid_targets)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Weighted average based on number of valid samples
        batch_samples = mask.sum().item()
        running_loss += loss.item() * batch_samples
        total_samples += batch_samples

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    return epoch_loss


def validate_epoch(model, dataloader, criterion, device):
    """Runs validation and calculates metrics."""
    model.eval()
    running_loss = 0.0
    total_samples = 0

    all_errors = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)

            # Metadata for reconstruction
            baselines = batch["baseline"]  # List of (T, 2) arrays
            timestamps = batch["timestamps"]  # List of (T,) arrays

            outputs = model(features)  # (B, 2, T)

            # 1. Calculate Loss
            mask_expanded = mask.unsqueeze(1).expand_as(targets)
            if mask_expanded.sum() > 0:
                valid_outputs = outputs[mask_expanded]
                valid_targets = targets[mask_expanded]
                loss = criterion(valid_outputs, valid_targets)

                batch_samples = mask.sum().item()
                running_loss += loss.item() * batch_samples
                total_samples += batch_samples

            # 2. Calculate Distance Metrics
            # Iterate through batch items to handle variable lengths correctly
            batch_size = features.size(0)
            for i in range(batch_size):
                # Get valid length from timestamps or baseline
                t_len = len(timestamps[i])

                # Slice prediction to actual length: (2, T) -> (T, 2)
                pred_offsets = outputs[i, :, :t_len].cpu().numpy().T

                # Get Baseline: (T, 2) -> (Lat, Lon)
                base_pos = baselines[i]

                # Reconstruct Predictions
                pred_lat, pred_lon = cartesian_to_wgs84(
                    pred_offsets[:, 0],
                    pred_offsets[:, 1],
                    base_pos[:, 0],
                    base_pos[:, 1],
                )

                # Get Ground Truth (if available in batch)
                # We need to reconstruct GT from targets to be consistent or use stored GT if passed
                # The dataset doesn't pass raw GT coords directly in batch dict key "gt_coords" in collate,
                # but we can reconstruct from target offsets + baseline for validation.

                target_offsets = targets[i, :, :t_len].cpu().numpy().T

                # Only evaluate on valid mask
                valid_mask = mask[i, :t_len].cpu().numpy().astype(bool)

                if valid_mask.sum() > 0:
                    # Reconstruct GT Lat/Lon
                    gt_lat, gt_lon = cartesian_to_wgs84(
                        target_offsets[valid_mask, 0],
                        target_offsets[valid_mask, 1],
                        base_pos[valid_mask, 0],
                        base_pos[valid_mask, 1],
                    )

                    p_lat = pred_lat[valid_mask]
                    p_lon = pred_lon[valid_mask]

                    # Haversine Distance
                    dists = haversine_distance(gt_lat, gt_lon, p_lat, p_lon)
                    all_errors.extend(dists)

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0

    if len(all_errors) > 0:
        mean_dist = np.mean(all_errors)
        p50 = np.percentile(all_errors, 50)
        p95 = np.percentile(all_errors, 95)
        score = (p50 + p95) / 2.0
    else:
        mean_dist = 0.0
        score = 0.0

    return epoch_loss, mean_dist, score


def train_model():
    """Main training loop."""
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Load Data
    print("Loading data...")
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=True)

    # Initialize Model
    model = HybridResUNetGRU().to(device)

    # Optimizer & Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    criterion = nn.L1Loss()

    best_score = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print("Starting training...")
    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dist, val_score = validate_epoch(
            model, val_loader, criterion, device
        )

        scheduler.step(val_score)

        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")
        print(f"  Train Loss (MAE m): {train_loss:.6f}")
        print(f"  Val Loss (MAE m):   {val_loss:.6f}")
        print(f"  Val Mean Dist (m):  {val_dist:.6f}")
        print(f"  Val Score (50/95):  {val_score:.6f}")

        # Early Stopping based on Competition Score
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved!")
        else:
            patience_counter += 1
            print(f"  -> Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}")

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Score: {best_score:.6f}")


def generate_submission():
    """Generates submission file for the test set."""
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Load Data
    _, _, test_loader = get_dataloaders(load_cached_data=True)

    # Load Model
    model = HybridResUNetGRU().to(device)
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(best_model_path):
        print("No trained model found. Cannot generate submission.")
        return

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    results = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)

            # Metadata
            baselines = batch["baseline"]
            timestamps = batch["timestamps"]
            drive_ids = batch["drive_id"]
            phone_names = batch["phone_name"]

            # Inference
            outputs = model(features)  # (B, 2, T)

            batch_size = features.size(0)
            for i in range(batch_size):
                t_len = len(timestamps[i])

                # Get predictions (North, East)
                pred_offsets = outputs[i, :, :t_len].cpu().numpy().T

                # Get Baseline (Lat, Lon)
                base_pos = baselines[i]

                # Reconstruct WGS84
                pred_lat, pred_lon = cartesian_to_wgs84(
                    pred_offsets[:, 0],
                    pred_offsets[:, 1],
                    base_pos[:, 0],
                    base_pos[:, 1],
                )

                # Prepare rows for DataFrame
                # tripId format: drive_id + "-" + phone_name
                trip_id = f"{drive_ids[i]}-{phone_names[i]}"

                for t, lat, lon in zip(timestamps[i], pred_lat, pred_lon):
                    results.append(
                        {
                            "tripId": trip_id,
                            "UnixTimeMillis": t,
                            "LatitudeDegrees": lat,
                            "LongitudeDegrees": lon,
                        }
                    )

    # Create DataFrame
    submission_df = pd.DataFrame(results)

    # Ensure columns are in correct order
    cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    submission_df = submission_df[cols]

    # Save
    out_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(out_path, index=False)
    print(f"Submission saved to {out_path}")
    print(f"Total predictions: {len(submission_df)}")


if __name__ == "__main__":
    train_model()
    generate_submission()
