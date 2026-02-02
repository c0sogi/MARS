import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.train import train_model
from library.inference import generate_submission
from library.data_loader import load_data, process_trip, create_sliding_windows
from library.model import IMULocalTrajectoryCNN


def run_pipeline():
    # -------------------------------------------------------------------------
    # 1. Train Model (Fast Baseline)
    # -------------------------------------------------------------------------
    print("Step 1: Training Model (Fast Baseline)...")
    # Using 50% of data and forcing reprocessing to ensure sampling is applied
    train_model(debug_sample_fraction=0.5, load_cached_data=False)

    # -------------------------------------------------------------------------
    # 2. Validation & Metric Calculation
    # -------------------------------------------------------------------------
    print("\nStep 2: Validation and Metric Calculation...")

    # Load validation data (tensors)
    # We load cached data here because train_model just generated it (or we rely on what's there)
    # Note: train_model saves to cache. We can load from cache now.
    _, val_dataset = load_data(mode="train", load_cached_data=True)

    # Load Metadata to reconstruct Phone mappings
    val_meta_path = Config.VAL_METADATA_PATH
    if not os.path.exists(val_meta_path):
        print("Validation metadata not found. Cannot compute metric.")
        return

    val_meta_df = pd.read_csv(val_meta_path)
    unique_trips = val_meta_df["tripId"].unique()

    # Reconstruct mapping: Sample Index -> Phone Name
    # We must replicate the data loading logic to ensure alignment
    print("Reconstructing validation metadata alignment...")
    sample_phone_map = []

    # The load_data function processes trips in the order they appear in unique_trips of the metadata file
    # We must iterate in the exact same order
    for trip in unique_trips:
        trip_info = val_meta_df[val_meta_df["tripId"] == trip].iloc[0]
        phone_name = trip_info["phone_name"]

        # Process trip to get the dataframe length
        # We don't need the full heavy processing, just the length after cleaning
        # However, process_trip does data cleaning (dropping NaNs). We must run it.
        df_trip = process_trip(
            trip, trip_info["gnss_path"], trip_info["imu_path"], val_meta_df
        )

        if df_trip is not None and len(df_trip) > Config.WINDOW_SIZE:
            # The sliding window logic reduces the number of samples
            # n_samples = len(df) - window_size + 1 (if we iterate range(half, n-half))
            # Logic in create_sliding_windows: range(half_window, n_samples - half_window)
            # Count = (n_samples - half_window) - half_window = n_samples - 2*half_window
            # Since window_size is odd (11), 2*half_window = 10. n_samples - 10.
            # Let's verify exact count logic from create_sliding_windows
            half_window = Config.WINDOW_SIZE // 2
            n_samples = len(df_trip)
            num_windows = len(range(half_window, n_samples - half_window))

            sample_phone_map.extend([phone_name] * num_windows)

    # Verify alignment
    if len(sample_phone_map) != len(val_dataset):
        print(
            f"Warning: Metadata alignment mismatch. Meta: {len(sample_phone_map)}, Dataset: {len(val_dataset)}"
        )
        # If mismatch, we cannot compute per-phone metric accurately.
        # Fallback: Treat as single phone or abort.
        # For this baseline, we will proceed with truncation if meta is larger, or error if smaller.
        if len(sample_phone_map) > len(val_dataset):
            sample_phone_map = sample_phone_map[: len(val_dataset)]
        else:
            print("Critical Error: Dataset larger than metadata map.")
            return

    # Run Inference
    device = torch.device(Config.DEVICE)
    model = IMULocalTrajectoryCNN(
        input_dim=Config.INPUT_DIM,
        window_size=Config.WINDOW_SIZE,
        output_dim=Config.OUTPUT_DIM,
        cnn_channels=Config.CNN_CHANNELS,
        kernel_size=Config.CNN_KERNEL_SIZE,
        cnn_dropout=Config.CNN_DROPOUT,
        mlp_hidden_dims=Config.MLP_HIDDEN_DIMS,
        mlp_dropout=Config.MLP_DROPOUT,
    )

    if not os.path.exists(Config.MODEL_PATH):
        print("Model file not found.")
        return

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    all_preds = []
    all_targets = []
    all_inputs = []  # Keep for failure analysis

    print("Running validation inference...")
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())
            # Store mean of input features across window for analysis (Batch, Features)
            # Input shape: (Batch, Window, Features) -> Mean over window -> (Batch, Features)
            all_inputs.append(inputs.mean(dim=1).cpu().numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    X_mean = np.concatenate(all_inputs, axis=0)

    # Compute Errors (Euclidean Distance in Meters)
    # y is [DeltaEast, DeltaNorth]
    errors = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))

    # Compute Metric
    df_scores = pd.DataFrame({"phone": sample_phone_map, "error": errors})

    def compute_phone_score(group):
        p50 = np.percentile(group["error"], 50)
        p95 = np.percentile(group["error"], 95)
        return (p50 + p95) / 2

    phone_scores = df_scores.groupby("phone").apply(compute_phone_score)
    final_metric = phone_scores.mean()

    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 3. Failure Analysis
    # -------------------------------------------------------------------------
    print("\nStep 3: Failure Analysis...")
    # Correlate errors with input features
    # Config.FEATURE_NAMES gives us feature names
    feature_names = Config.FEATURE_NAMES

    correlations = {}
    for i, feat_name in enumerate(feature_names):
        # Calculate Pearson correlation between feature value and error
        corr, _ = pearsonr(X_mean[:, i], errors)
        correlations[feat_name] = corr

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error Magnitude:")
    for name, corr in sorted_corr[:5]:
        print(f"  {name}: {corr:.4f}")

    # -------------------------------------------------------------------------
    # 4. Submission Generation
    # -------------------------------------------------------------------------
    THRESHOLD = 4.256982128481356
    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric:.6f}) is below threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(load_cached_data=False)  # Reprocess test data to be safe
    else:
        print(
            f"\nMetric ({final_metric:.6f}) is NOT below threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    run_pipeline()
