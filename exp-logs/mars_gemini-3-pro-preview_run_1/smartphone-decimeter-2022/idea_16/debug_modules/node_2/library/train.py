import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import set_seed, WGS84Utils
from library.data_loader import load_dataset, GNSSSequenceDataset
from library.model import ResUNet1D


class MultiScaleMAELoss(nn.Module):
    """
    Computes MAE loss across multiple temporal resolutions for Deep Supervision.
    Dynamically downsamples ground truth targets to match prediction shapes.
    """

    def __init__(self, weights):
        super(MultiScaleMAELoss, self).__init__()
        self.weights = weights
        self.mae = nn.L1Loss(reduction="none")

    def forward(self, predictions, targets, mask):
        """
        Args:
            predictions: List of tensors [Head0, Head1, Head2, ...]
                         Head 0 is full resolution.
            targets: Tensor of shape (B, C, L) - Full resolution ground truth.
            mask: Tensor of shape (B, L) - Valid data mask.
        """
        total_loss = 0.0

        # predictions is a list of outputs from the model heads
        # We assume they are ordered by key in the model: 0, 1, 2...
        # where 0 is full res, 1 is half res, etc.

        for i, pred in enumerate(predictions):
            weight = self.weights.get(i, 0.0)
            if weight == 0.0:
                continue

            # Determine scale factor based on temporal dimension
            scale_factor = targets.shape[-1] // pred.shape[-1]

            # Downsample targets and mask if necessary
            if scale_factor > 1:
                # Average pooling for regression targets
                curr_targets = F.avg_pool1d(
                    targets, kernel_size=scale_factor, stride=scale_factor
                )
                # Nearest neighbor for mask to keep it binary-ish (or avg and threshold)
                # Using interpolate nearest is safer for mask
                curr_mask = F.interpolate(
                    mask.unsqueeze(1), size=pred.shape[-1], mode="nearest"
                ).squeeze(1)
            else:
                curr_targets = targets
                curr_mask = mask

            # Compute masked MAE
            loss = self.mae(pred, curr_targets)
            # Loss shape: (B, C, L_scaled)

            # Apply mask (expand to C channels)
            mask_expanded = curr_mask.unsqueeze(1).expand_as(loss)

            masked_loss = (loss * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)

            total_loss += weight * masked_loss

        return total_loss


def train_model(load_cached_data=True):
    """
    Main training loop.
    """
    set_seed(Config.SEED)

    # 1. Load Data
    train_df = load_dataset(
        "train", load_cached_data=load_cached_data, debug=Config.DEBUG
    )
    val_df = load_dataset("val", load_cached_data=load_cached_data, debug=Config.DEBUG)

    train_dataset = GNSSSequenceDataset(train_df, mode="train")
    val_dataset = GNSSSequenceDataset(val_df, mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Setup Model & Optimization
    device = torch.device(Config.DEVICE)
    model = ResUNet1D().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    criterion = MultiScaleMAELoss(Config.DEEP_SUPERVISION_WEIGHTS)

    # 3. Training Loop
    best_val_metric = float("inf")
    patience_counter = 0

    print(f"Starting training on {device} for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss_sum = 0.0
        train_steps = 0

        start_time = time.time()

        for batch in train_loader:
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()

            outputs = model(features)
            loss = criterion(outputs, targets, mask)

            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), Config.GRAD_CLIP_MAX_NORM
            )

            optimizer.step()

            train_loss_sum += loss.item()
            train_steps += 1

        avg_train_loss = train_loss_sum / train_steps

        # 4. Validation
        model.eval()
        val_loss_sum = 0.0
        val_steps = 0
        all_errors = []

        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                targets = batch["targets"].to(device)
                mask = batch["mask"].to(device)

                outputs = model(features)
                loss = criterion(outputs, targets, mask)

                val_loss_sum += loss.item()
                val_steps += 1

                # Metric Calculation (using Head 0 - Full Resolution)
                # Targets are (dNorth, dEast) in meters
                pred_full = outputs[0]  # (B, 2, L)

                # Calculate Euclidean distance error per point
                # Shape: (B, L)
                dist_errors = torch.sqrt(torch.sum((pred_full - targets) ** 2, dim=1))

                # Apply mask to select valid points
                valid_errors = dist_errors[mask == 1.0]
                all_errors.append(valid_errors.cpu().numpy())

        avg_val_loss = val_loss_sum / val_steps

        # Compute Metric (Mean of 50th and 95th percentiles)
        flat_errors = np.concatenate(all_errors)
        p50 = np.percentile(flat_errors, 50)
        p95 = np.percentile(flat_errors, 95)
        val_metric = (p50 + p95) / 2

        scheduler.step()

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Time: {elapsed:.0f}s | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f} | "
            f"Val Metric (50/95 avg): {val_metric:.6f}"
        )

        # Checkpoint
        if val_metric < best_val_metric:
            best_val_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  >>> Best model saved (Metric: {best_val_metric:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("  >>> Early stopping triggered.")
                break


def generate_submission(load_cached_data=True):
    """
    Generates submission file using the trained model.
    """
    set_seed(Config.SEED)

    # Load Test Data
    test_df = load_dataset(
        "test", load_cached_data=load_cached_data, debug=Config.DEBUG
    )
    test_dataset = GNSSSequenceDataset(test_df, mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Load Model
    device = torch.device(Config.DEVICE)
    model = ResUNet1D().to(device)

    if not os.path.exists(Config.MODEL_PATH):
        print("Model checkpoint not found. Cannot generate submission.")
        return

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    print("Generating predictions...")

    results = []

    # Get metadata from dataset to map back to tripId
    # GNSSSequenceDataset stores metadata list: [(drive_id, phone_name), ...]
    seq_metadata = test_dataset.metadata

    seq_idx = 0
    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            baselines = batch["baselines"].numpy()  # (B, L, 3)
            timestamps = batch["timestamps"].numpy()  # (B, L)

            # Forward pass
            outputs = model(features)
            pred_res = outputs[0]  # Head 0 (B, 2, L) -> (dNorth, dEast)

            # Convert to numpy (B, L, 2)
            pred_res = pred_res.permute(0, 2, 1).cpu().numpy()

            batch_size = features.shape[0]

            for b in range(batch_size):
                drive_id, phone_name = seq_metadata[seq_idx]
                seq_idx += 1

                # Get valid length from mask
                # Note: mask is 1.0 for valid, 0.0 for padding
                # We can sum the mask to get length
                valid_len = int(mask[b].sum().item())

                # Slice valid data
                valid_preds = pred_res[b, :valid_len, :]  # (L_valid, 2) -> (dN, dE)
                valid_base = baselines[b, :valid_len, :]  # (L_valid, 3) -> (X, Y, Z)
                valid_time = timestamps[b, :valid_len]  # (L_valid)

                # Reconstruct Coordinates
                # 1. Convert Baseline ECEF to Geodetic (Lat, Lon, Alt)
                # Vectorized conversion is preferred but utils are scalar/numpy based
                # We iterate or use numpy vectorization if implemented.
                # WGS84Utils methods use numpy, so they support arrays.

                wls_x = valid_base[:, 0]
                wls_y = valid_base[:, 1]
                wls_z = valid_base[:, 2]

                ref_lat, ref_lon, ref_alt = WGS84Utils.ecef_to_geodetic(
                    wls_x, wls_y, wls_z
                )

                # 2. Convert Predicted ENU offsets to ECEF
                # Preds are (dNorth, dEast). dUp is assumed 0.
                # Utils.enu_to_ecef takes (e, n, u, ref...)
                d_n = valid_preds[:, 0]
                d_e = valid_preds[:, 1]
                d_u = np.zeros_like(d_n)

                pred_x, pred_y, pred_z = WGS84Utils.enu_to_ecef(
                    d_e, d_n, d_u, ref_lat, ref_lon, ref_alt
                )

                # 3. Convert Predicted ECEF back to Geodetic
                final_lat, final_lon, _ = WGS84Utils.ecef_to_geodetic(
                    pred_x, pred_y, pred_z
                )

                # Create DataFrame for this sequence
                trip_id = f"{drive_id}-{phone_name}"

                seq_df = pd.DataFrame(
                    {
                        "tripId": trip_id,
                        "UnixTimeMillis": valid_time,
                        "LatitudeDegrees": final_lat,
                        "LongitudeDegrees": final_lon,
                    }
                )

                results.append(seq_df)

    # Concatenate all results
    submission_df = pd.concat(results, ignore_index=True)

    # The submission file requires specific rows corresponding to sample_submission.csv
    # We need to filter/join with sample_submission to ensure exact format
    sample_sub = pd.read_csv(os.path.join(Config.INPUT_DIR, "sample_submission.csv"))

    # Ensure types match for merge
    submission_df["UnixTimeMillis"] = submission_df["UnixTimeMillis"].astype(np.int64)
    sample_sub["UnixTimeMillis"] = sample_sub["UnixTimeMillis"].astype(np.int64)

    # Merge to keep only required rows and correct order
    # We left join sample_sub onto our predictions
    final_sub = pd.merge(
        sample_sub[["tripId", "UnixTimeMillis"]],
        submission_df,
        on=["tripId", "UnixTimeMillis"],
        how="left",
    )

    # Fill missing (if any) with WLS baseline or interpolation?
    # Our pipeline processes all test data found in metadata.
    # If timestamps align (1Hz rounding), it should match.
    # If missing, simple fill with previous value or linear interpolation is standard,
    # but let's assume coverage is good.

    # Check for NaNs
    if final_sub.isnull().any().any():
        print("Warning: NaNs in submission. Filling with forward fill.")
        final_sub = final_sub.fillna(method="ffill").fillna(method="bfill")

    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    final_sub.to_csv(Config.SUBMISSION_PATH, index=False)
