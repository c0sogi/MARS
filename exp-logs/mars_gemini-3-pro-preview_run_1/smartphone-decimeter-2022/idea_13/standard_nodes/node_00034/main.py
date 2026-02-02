import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, haversine_distance, WGS84Converter
from library.data_processing import GNSSPreprocessor
from library.dataset import GNSSSequenceDataset
from library.model import MultiScaleResUNet1D
from library.loss import DeepSupervisionMAELoss
from library.train import Trainer


def calculate_competition_metric(df_val, preds_lat, preds_lon):
    """
    Calculates the competition metric: mean of the 50th and 95th percentile distance errors,
    averaged across phones.
    """
    # Create a working copy
    df = df_val.copy()
    df["pred_lat"] = preds_lat
    df["pred_lon"] = preds_lon

    # Calculate Haversine distance
    df["dist_error"] = haversine_distance(
        df["LatitudeDegrees"].values,
        df["LongitudeDegrees"].values,
        df["pred_lat"].values,
        df["pred_lon"].values,
    )

    # Create a unique phone identifier
    df["phone_id"] = df["drive_id"] + "_" + df["phone_name"]

    phone_scores = []
    for phone_id, group in df.groupby("phone_id"):
        errors = group["dist_error"].values
        p50 = np.percentile(errors, 50)
        p95 = np.percentile(errors, 95)
        phone_scores.append((p50 + p95) / 2)

    final_metric = np.mean(phone_scores)
    return final_metric, df["dist_error"].values


def perform_failure_analysis(val_loader, model, device, feature_names):
    """
    Correlates prediction errors with input features to identify failure modes.
    """
    model.eval()
    all_errors = []
    all_features = []

    # We need to compute errors per point first
    # This requires running inference again or restructuring, but since we need features aligned with errors,
    # we'll run a pass to collect features and targets, then compute errors.

    # However, we already computed global errors in calculate_competition_metric.
    # To map them back to features, we need to ensure alignment.
    # The validation loader yields sequences. We need to flatten them.

    converter = WGS84Converter()

    flat_features = []
    flat_errors = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)  # (B, C, T)

            # Predict
            preds = model(features)  # (B, 2, T) -> d_east, d_north
            preds = preds.cpu().numpy()

            # Metadata for reconstruction
            wls_lats = batch["wls_lat"].numpy()
            wls_lons = batch["wls_lon"].numpy()
            timestamps = batch["UnixTimeMillis"].numpy()

            # Ground Truth for error calculation
            # Note: The dataset might pad. We need to mask padding.
            # But for failure analysis on validation, we want valid points.
            # The dataset provides 'targets' (d_east, d_north GT) if available.
            # Let's use the provided GT targets to compute error in meters directly for correlation.

            gt_targets = batch["targets"].numpy()  # (B, 2, T)

            # Mask
            mask = batch["mask"].numpy()  # (B, T)

            # Flatten
            B, C, T = features.shape

            # Features: (B, C, T) -> (B*T, C)
            feat_flat = features.permute(0, 2, 1).reshape(-1, C).cpu().numpy()

            # Preds: (B, 2, T) -> (B*T, 2)
            pred_flat = preds.transpose(0, 2, 1).reshape(-1, 2)

            # GT: (B, 2, T) -> (B*T, 2)
            gt_flat = gt_targets.transpose(0, 2, 1).reshape(-1, 2)

            # Mask: (B, T) -> (B*T)
            mask_flat = mask.reshape(-1).astype(bool)

            # Filter valid
            feat_valid = feat_flat[mask_flat]
            pred_valid = pred_flat[mask_flat]
            gt_valid = gt_flat[mask_flat]

            # Calculate Euclidean distance error in meters (approximation is fine for correlation)
            # Error = sqrt((pred_e - gt_e)^2 + (pred_n - gt_n)^2)
            errors = np.sqrt(np.sum((pred_valid - gt_valid) ** 2, axis=1))

            flat_features.append(feat_valid)
            flat_errors.append(errors)

    all_features = np.concatenate(flat_features, axis=0)
    all_errors = np.concatenate(flat_errors, axis=0)

    # Compute correlations
    print("\n--- Failure Analysis: Feature vs Error Correlation ---")
    correlations = {}
    for i, feat_name in enumerate(feature_names):
        if i < all_features.shape[1]:
            feat_vals = all_features[:, i]
            # Handle constant features
            if np.std(feat_vals) == 0:
                corr = 0.0
            else:
                corr = np.corrcoef(feat_vals, all_errors)[0, 1]
            correlations[feat_name] = corr

    # Sort and print top correlations
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, corr in sorted_corr[:10]:
        print(f"{name}: {corr:.4f}")


def run_pipeline():
    # 1. Configuration Overrides for Fast Baseline
    Config.NUM_EPOCHS = 5
    Config.TRAIN_WINDOW_SIZE = 128
    Config.BATCH_SIZE = 64

    print(
        f"Running pipeline with: Epochs={Config.NUM_EPOCHS}, Window={Config.TRAIN_WINDOW_SIZE}, Batch={Config.BATCH_SIZE}"
    )

    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data Processing
    preprocessor = GNSSPreprocessor()

    # Load Train/Val Data (Cached if available)
    train_df = preprocessor.get_train_data(load_cached_data=True)
    val_df = preprocessor.get_val_data(load_cached_data=True)

    feature_cols = Config.FEATURE_NAMES
    target_cols = ["d_east", "d_north"]

    # 3. Datasets & Loaders
    print("Initializing Datasets...")
    train_dataset = GNSSSequenceDataset(
        train_df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        window_size=Config.TRAIN_WINDOW_SIZE,
        stride=Config.TRAIN_WINDOW_SIZE // 2,  # Overlap for training
        mode="train",
    )

    val_dataset = GNSSSequenceDataset(
        val_df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        window_size=Config.TRAIN_WINDOW_SIZE,
        stride=Config.TRAIN_WINDOW_SIZE,  # No overlap for validation
        mode="val",
        stats=train_dataset.stats,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # 4. Model Setup
    model = MultiScaleResUNet1D().to(device)
    criterion = DeepSupervisionMAELoss()
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # 5. Training
    trainer = Trainer(model, criterion, optimizer, scheduler, device, checkpoint_path)
    trainer.fit(
        train_loader,
        val_loader,
        epochs=Config.NUM_EPOCHS,
        patience=Config.EARLY_STOPPING_PATIENCE,
    )

    # 6. Validation Assessment & Metric Calculation
    print("\n--- Performing Validation Assessment ---")
    # We need to reconstruct the full validation set predictions to calculate the metric per phone
    # The val_loader yields windows. We need to stitch them or just process them and map back to the dataframe.
    # Since we used non-overlapping stride=window_size, we can just iterate and collect.

    model.eval()
    converter = WGS84Converter()

    val_preds_lat = []
    val_preds_lon = []
    val_timestamps = []
    val_trip_ids = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)

            # Forward
            preds = model(features).cpu().numpy()  # (B, 2, T)

            # Metadata
            b_trip_ids = batch["trip_id"]
            b_timestamps = batch["UnixTimeMillis"].numpy()
            b_wls_lat = batch["wls_lat"].numpy()
            b_wls_lon = batch["wls_lon"].numpy()
            b_seq_len = batch["seq_len"].numpy()

            for i in range(features.shape[0]):
                length = b_seq_len[i]

                # Extract valid sequence
                d_east = preds[i, 0, :length]
                d_north = preds[i, 1, :length]
                ref_lat = b_wls_lat[i, :length]
                ref_lon = b_wls_lon[i, :length]
                ts = b_timestamps[i, :length]
                tid = b_trip_ids[i]

                # Convert to deg
                p_lat, p_lon = converter.meters_to_deg(
                    d_east, d_north, ref_lat, ref_lon
                )

                val_preds_lat.extend(p_lat)
                val_preds_lon.extend(p_lon)
                val_timestamps.extend(ts)
                val_trip_ids.extend([tid] * length)

    # Create prediction dataframe
    df_val_pred = pd.DataFrame(
        {
            "tripId": val_trip_ids,
            "UnixTimeMillis": val_timestamps,
            "pred_lat": val_preds_lat,
            "pred_lon": val_preds_lon,
        }
    )

    # Merge with original validation dataframe to get ground truth
    # val_df has 'drive_id', 'phone_name', 'UnixTimeMillis', 'LatitudeDegrees', 'LongitudeDegrees'
    # Construct tripId in val_df
    val_df["tripId"] = val_df["drive_id"] + "-" + val_df["phone_name"]

    # Merge on tripId and Time
    # Note: Timestamps in val_df might be slightly different if not aligned, but preprocessor aligns them.
    df_eval = pd.merge(
        val_df, df_val_pred, on=["tripId", "UnixTimeMillis"], how="inner"
    )

    metric, _ = calculate_competition_metric(
        df_eval, df_eval["pred_lat"], df_eval["pred_lon"]
    )
    print(f"Final Validation Metric: {metric}")

    # 7. Failure Analysis
    perform_failure_analysis(val_loader, model, device, feature_cols)

    # 8. Submission Generation
    threshold = 3.802240262877392
    if metric < threshold:
        print(
            f"\nValidation metric ({metric}) is better than threshold ({threshold}). Generating submission..."
        )

        # Get Test Loader
        test_loader = get_test_dataloader(train_dataset.stats, debug=False)

        submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        trainer.predict(test_loader, submission_path)
    else:
        print(
            f"\nValidation metric ({metric}) did not beat threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    run_pipeline()
