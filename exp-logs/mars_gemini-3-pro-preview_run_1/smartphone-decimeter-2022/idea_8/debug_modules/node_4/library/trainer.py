import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.model import CascadedResUNet
from library.data_loader import (
    get_train_val_loaders,
    process_drive,
    GnssSequenceDataset,
)
from library.utils import enu_to_llh, haversine_distance


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(load_cached_data=True):
    set_seed(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Data
    print("Loading data...")
    train_loader, val_loader = get_train_val_loaders(load_cached_data=load_cached_data)

    # Initialize Model
    model = CascadedResUNet().to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # Loss Function (L1 Loss for MAE)
    criterion = nn.L1Loss()

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(Config.EPOCHS):
        # --- Training ---
        model.train()
        train_loss = 0.0

        for batch_idx, (features, targets, _, _) in enumerate(train_loader):
            features = features.to(device)
            targets = targets.to(device)  # Shape: (B, 2, L)

            optimizer.zero_grad()

            # Forward
            out1, final_out = model(features)

            # Deep Supervision Loss
            loss1 = criterion(out1, targets)
            loss_final = criterion(final_out, targets)
            loss = (Config.LOSS_WEIGHT_STAGE1 * loss1) + (
                Config.LOSS_WEIGHT_FINAL * loss_final
            )

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        all_errors = []

        with torch.no_grad():
            for features, targets, wls, meta in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                wls = wls.numpy()  # (B, L, 3)

                _, final_out = model(features)

                # Calculate Loss
                loss_final = criterion(final_out, targets)
                val_loss += loss_final.item()

                # Calculate Metric (Distance Error)
                pred_enu = final_out.cpu().numpy()  # (B, 2, L)
                target_enu = targets.cpu().numpy()

                # Iterate over batch to compute Haversine distances
                for b in range(pred_enu.shape[0]):
                    wls_b = wls[b]  # (L, 3)

                    pred_e = pred_enu[b, 0, :]
                    pred_n = pred_enu[b, 1, :]

                    target_e = target_enu[b, 0, :]
                    target_n = target_enu[b, 1, :]

                    ref_lat = wls_b[:, 0]
                    ref_lon = wls_b[:, 1]
                    ref_alt = wls_b[:, 2]

                    # Reconstruct Predicted Locations
                    pred_lat, pred_lon, _ = enu_to_llh(
                        pred_e, pred_n, np.zeros_like(pred_e), ref_lat, ref_lon, ref_alt
                    )

                    # Reconstruct Ground Truth Locations (from targets)
                    gt_lat, gt_lon, _ = enu_to_llh(
                        target_e,
                        target_n,
                        np.zeros_like(target_e),
                        ref_lat,
                        ref_lon,
                        ref_alt,
                    )

                    # Calculate Distance
                    dists = haversine_distance(pred_lat, pred_lon, gt_lat, gt_lon)

                    # Filter out padding (where ref_lat is 0 or similar check if needed,
                    # but GnssSequenceDataset pads WLS with edge values, so valid calculation happens everywhere.
                    # We accept the metric on padded data as a proxy or assume validation set is clean enough).
                    all_errors.extend(dists)

        avg_val_loss = val_loss / len(val_loader)

        # Calculate Competition Metric
        if all_errors:
            errors = np.array(all_errors)
            p50 = np.percentile(errors, 50)
            p95 = np.percentile(errors, 95)
            score = (p50 + p95) / 2
        else:
            score = 0.0

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {avg_train_loss:.10f} | Val Loss: {avg_val_loss:.10f} | Val Score (50/95): {score:.10f}"
        )

        scheduler.step(avg_val_loss)

        # Early Stopping & Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "./working/model_weights.pth")
            print("  -> Model saved.")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break


def generate_submission(load_cached_data=True):
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    model = CascadedResUNet().to(device)
    weights_path = "./working/model_weights.pth"
    if not os.path.exists(weights_path):
        print("Model weights not found. Skipping inference.")
        return

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # Load Test Metadata
    test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

    # Group by drive to process sequentially
    drive_groups = test_meta.groupby(["drive_id", "phone_name"])

    results = []

    print(f"Generating submission for {len(drive_groups)} drives...")

    for (drive_id, phone_name), group in drive_groups:
        gnss_path = group.iloc[0]["gnss_path"]

        # Process Drive
        df = process_drive(
            drive_id,
            phone_name,
            gnss_path,
            gt_df=None,
            load_cached_data=load_cached_data,
        )

        if df.empty:
            # Fallback if processing fails: fill with 0s or NaNs, but better to skip and let merge handle it
            continue

        # Create Dataset/Loader for this single drive
        dataset = GnssSequenceDataset(
            [df], sequence_length=Config.SEQUENCE_LENGTH, mode="test"
        )
        loader = DataLoader(
            dataset, batch_size=32, shuffle=False, num_workers=Config.NUM_WORKERS
        )

        drive_preds = []

        with torch.no_grad():
            for features, _, wls, meta in loader:
                features = features.to(device)
                wls = wls.numpy()
                meta = meta.numpy()

                _, final_out = model(features)
                pred_enu = final_out.cpu().numpy()  # (B, 2, L)

                # Iterate batch
                for b in range(pred_enu.shape[0]):
                    wls_b = wls[b]
                    meta_b = meta[b]
                    pred_e = pred_enu[b, 0, :]
                    pred_n = pred_enu[b, 1, :]

                    ref_lat = wls_b[:, 0]
                    ref_lon = wls_b[:, 1]
                    ref_alt = wls_b[:, 2]

                    pred_lat, pred_lon, _ = enu_to_llh(
                        pred_e, pred_n, np.zeros_like(pred_e), ref_lat, ref_lon, ref_alt
                    )

                    # Store results
                    for i in range(len(meta_b)):
                        ts = meta_b[i]
                        drive_preds.append(
                            {
                                "UnixTimeMillis": ts,
                                "LatitudeDegrees": pred_lat[i],
                                "LongitudeDegrees": pred_lon[i],
                            }
                        )

        if not drive_preds:
            continue

        # Aggregate predictions for this drive
        pred_df = pd.DataFrame(drive_preds)
        # Average predictions for overlapping timestamps (due to windowing)
        pred_df = pred_df.groupby("UnixTimeMillis", as_index=False).mean()

        # Merge with requested timestamps for this drive
        # Create a join key based on rounding logic used in feature engineering
        group["JoinKey"] = ((group["UnixTimeMillis"] + 500) // 1000 * 1000).astype(
            np.int64
        )
        pred_df["JoinKey"] = pred_df["UnixTimeMillis"]

        # Drop original UnixTimeMillis from pred_df to avoid collision
        pred_df = pred_df.drop(columns=["UnixTimeMillis"])

        merged = pd.merge(
            group, pred_df, on="JoinKey", how="left", suffixes=("", "_pred")
        )

        # Interpolate missing predictions (if any gaps)
        cols_to_interp = ["LatitudeDegrees_pred", "LongitudeDegrees_pred"]
        merged[cols_to_interp] = merged[cols_to_interp].interpolate(
            method="linear", limit_direction="both"
        )

        # Prepare final columns
        final_drive_df = merged[
            [
                "tripId",
                "UnixTimeMillis",
                "LatitudeDegrees_pred",
                "LongitudeDegrees_pred",
            ]
        ].copy()
        final_drive_df.rename(
            columns={
                "LatitudeDegrees_pred": "LatitudeDegrees",
                "LongitudeDegrees_pred": "LongitudeDegrees",
            },
            inplace=True,
        )

        results.append(final_drive_df)

    # Concatenate all
    if results:
        submission_df = pd.concat(results, ignore_index=True)

        # Ensure we cover all rows in sample_submission
        sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
        final_sub = pd.merge(
            sample_sub[["tripId", "UnixTimeMillis"]],
            submission_df,
            on=["tripId", "UnixTimeMillis"],
            how="left",
        )

        # Final fallback for any remaining NaNs (unlikely)
        final_sub = final_sub.interpolate(method="linear", limit_direction="both")

        # Save
        sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        final_sub.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")
    else:
        print("No results generated.")
