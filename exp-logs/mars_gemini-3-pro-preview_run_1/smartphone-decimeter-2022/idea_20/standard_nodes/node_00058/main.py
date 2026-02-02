import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import scipy.stats

# Import library modules
from library.config import CFG, set_seed
from library.dataset import get_train_val_datasets, get_test_dataset
from library.model import ResUNet1D
from library.engine import train_model
from library.utils import enu_to_wgs84, haversine_distance


def main():
    # 1. Setup
    # Override Config for Fast Baseline
    CFG.EPOCHS = 5  # Limit epochs to ensure completion within time limit
    CFG.WORKING_DIR = (
        "./working"  # Use main working dir to potentially hit existing caches if any
    )
    # Update paths in CFG based on the working dir change if necessary,
    # but the library uses CFG.WORKING_DIR to construct paths.
    # Re-setting paths just in case the class definition didn't pick up the change dynamically (it won't)
    CFG.TRAIN_CACHE_PATH = os.path.join(CFG.WORKING_DIR, "train_processed.parquet")
    CFG.VAL_CACHE_PATH = os.path.join(CFG.WORKING_DIR, "val_processed.parquet")
    CFG.TEST_CACHE_PATH = os.path.join(CFG.WORKING_DIR, "test_processed.parquet")
    CFG.BEST_MODEL_PATH = os.path.join(CFG.WORKING_DIR, "best_model.pth")

    set_seed(CFG.SEED)
    device = CFG.DEVICE

    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading Datasets...")
    # Attempt to load cached data to save time
    train_dataset, val_dataset, scaler = get_train_val_datasets(
        load_cached_data=True, debug=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.BATCH_SIZE,
        shuffle=True,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = ResUNet1D()

    # 4. Training
    print("Starting Training...")
    model = train_model(model, train_loader, val_loader)

    # 5. Validation & Failure Analysis
    print("Starting Validation and Failure Analysis...")
    model.eval()

    # We need to map predictions back to the dataframe rows
    # Create a results dataframe initialized with the validation dataframe
    val_df = val_dataset.df.copy()
    val_df["pred_east"] = np.nan
    val_df["pred_north"] = np.nan

    # Store predictions in a dictionary {orig_index: (east, north)}
    predictions_map = {}

    with torch.no_grad():
        for features, targets, mask, indices in val_loader:
            features = features.to(device)

            # Forward pass
            outputs = model(features)
            main_output = outputs[0]  # (B, 2, L)

            # Convert to numpy
            preds = main_output.cpu().numpy()  # (B, 2, L)
            mask_np = mask.numpy()  # (B, L)
            indices_np = indices.numpy()  # (B, L)

            # Iterate through batch
            for b in range(preds.shape[0]):
                valid_len = int(mask_np[b].sum())
                # Get valid predictions and indices
                # Note: indices are padded with -1
                valid_indices = indices_np[b, :valid_len]
                valid_preds = preds[b, :, :valid_len]  # (2, valid_len)

                for i, idx in enumerate(valid_indices):
                    if idx != -1:
                        predictions_map[idx] = (valid_preds[0, i], valid_preds[1, i])

    # Map predictions back to dataframe
    s_east = pd.Series({k: v[0] for k, v in predictions_map.items()})
    s_north = pd.Series({k: v[1] for k, v in predictions_map.items()})

    val_df["pred_east"] = val_df["orig_index"].map(s_east)
    val_df["pred_north"] = val_df["orig_index"].map(s_north)

    # Reconstruct Lat/Lon
    # WLS Lat/Lon are in val_df
    val_df["pred_lat"], val_df["pred_lon"] = enu_to_wgs84(
        val_df["pred_east"].values,
        val_df["pred_north"].values,
        val_df["WlsLatitudeDegrees"].values,
        val_df["WlsLongitudeDegrees"].values,
    )

    # Calculate Distance Error
    val_df["dist_error"] = haversine_distance(
        val_df["LatitudeDegrees"].values,
        val_df["LongitudeDegrees"].values,
        val_df["pred_lat"].values,
        val_df["pred_lon"].values,
    )

    # Calculate Metric
    # Mean of 50th and 95th percentile errors, averaged for each phone
    phone_scores = []
    for phone, group in val_df.groupby(["drive_id", "phone_name"]):
        errors = group["dist_error"].dropna().values
        if len(errors) == 0:
            continue
        p50 = np.percentile(errors, 50)
        p95 = np.percentile(errors, 95)
        phone_scores.append((p50 + p95) / 2)

    final_metric = np.mean(phone_scores)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation
    print("\nFailure Analysis (Correlation with Error Magnitude):")
    feature_cols = CFG.FEATURE_COLS
    correlations = {}
    for col in feature_cols:
        if col in val_df.columns:
            # Drop NaNs for correlation
            valid_data = val_df[[col, "dist_error"]].dropna()
            if len(valid_data) > 0:
                corr = valid_data[col].corr(valid_data["dist_error"])
                correlations[col] = corr

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, corr in sorted_corr[:5]:
        print(f"  {feat}: {corr:.4f}")

    # 6. Submission
    THRESHOLD = 3.802240262877392
    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) < Threshold ({THRESHOLD}). Generating Submission..."
        )

        test_dataset = get_test_dataset(scaler, load_cached_data=True, debug=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=CFG.BATCH_SIZE,
            shuffle=False,
            num_workers=CFG.NUM_WORKERS,
            pin_memory=True,
        )

        test_predictions_map = {}

        with torch.no_grad():
            for features, _, mask, indices in test_loader:
                features = features.to(device)
                outputs = model(features)
                main_output = outputs[0]

                preds = main_output.cpu().numpy()
                mask_np = mask.numpy()
                indices_np = indices.numpy()

                for b in range(preds.shape[0]):
                    valid_len = int(mask_np[b].sum())
                    valid_indices = indices_np[b, :valid_len]
                    valid_preds = preds[b, :, :valid_len]

                    for i, idx in enumerate(valid_indices):
                        if idx != -1:
                            test_predictions_map[idx] = (
                                valid_preds[0, i],
                                valid_preds[1, i],
                            )

        # Map to Test DF
        test_df = test_dataset.df.copy()
        s_east_test = pd.Series({k: v[0] for k, v in test_predictions_map.items()})
        s_north_test = pd.Series({k: v[1] for k, v in test_predictions_map.items()})

        test_df["pred_east"] = test_df["orig_index"].map(s_east_test)
        test_df["pred_north"] = test_df["orig_index"].map(s_north_test)

        # Reconstruct Lat/Lon
        test_df["pred_lat"], test_df["pred_lon"] = enu_to_wgs84(
            test_df["pred_east"].values,
            test_df["pred_north"].values,
            test_df["WlsLatitudeDegrees"].values,
            test_df["WlsLongitudeDegrees"].values,
        )

        # Prepare Submission File
        submission = pd.DataFrame(
            {
                "tripId": test_df["tripId"],
                "UnixTimeMillis": test_df["UnixTimeMillis"],
                "LatitudeDegrees": test_df["pred_lat"],
                "LongitudeDegrees": test_df["pred_lon"],
            }
        )

        # Fill NaNs if any with WLS baseline
        mask_nan = submission["LatitudeDegrees"].isna()
        if mask_nan.any():
            print(
                f"Warning: {mask_nan.sum()} NaN predictions found. Filling with WLS baseline."
            )
            submission.loc[mask_nan, "LatitudeDegrees"] = test_df.loc[
                mask_nan, "WlsLatitudeDegrees"
            ]
            submission.loc[mask_nan, "LongitudeDegrees"] = test_df.loc[
                mask_nan, "WlsLongitudeDegrees"
            ]

        submission.to_csv(CFG.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {CFG.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric ({final_metric}) >= Threshold ({THRESHOLD}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
