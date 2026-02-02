import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.data_loader import get_data, GNSSDataset
from library.model import WindowedMLP
from library.trainer import run_experiment
from library.utils import haversine_distance


def set_seed(seed=42):
    """Sets fixed random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_competition_metric(df):
    """
    Calculates the competition metric:
    Mean of ( (50th percentile error + 95th percentile error) / 2 ) across all trips.
    """
    # Calculate Haversine distance error for each prediction
    df["error"] = haversine_distance(
        df[Config.COL_LATITUDE], df[Config.COL_LONGITUDE], df["PredLat"], df["PredLon"]
    )

    # Calculate metric per trip (phone)
    trip_metrics = []
    # Group by tripId to calculate percentiles per phone trace
    for trip_id, group in df.groupby(Config.COL_TRIP_ID):
        errors = group["error"].values
        p50 = np.percentile(errors, 50)
        p95 = np.percentile(errors, 95)
        # Average of 50th and 95th percentiles for this phone
        trip_metrics.append((p50 + p95) / 2)

    # Mean across all phones (trips)
    final_metric = np.mean(trip_metrics)
    return final_metric


def perform_failure_analysis(df):
    """
    Correlates prediction error magnitude with input signal features.
    """
    # Features to analyze for correlation with error
    features = [Config.FEAT_CN0, Config.FEAT_UNC, Config.FEAT_SAT_COUNT]

    print("\nFailure Analysis (Spearman Correlation with Error):")
    print("-" * 50)
    for feat in features:
        if feat in df.columns:
            # Use Spearman correlation to capture monotonic relationships (robust to outliers)
            corr = df[feat].corr(df["error"], method="spearman")
            print(f"  {feat}: {corr:.4f}")


def main():
    # Ensure reproducibility
    set_seed(Config.RANDOM_STATE)

    print("==================================================")
    print("SMARTPHONE LOCATION PREDICTION PIPELINE")
    print("==================================================")

    # ---------------------------------------------------------
    # Phase 1: Training and Test Submission
    # ---------------------------------------------------------
    print("\n[Phase 1] Training Model and Generating Test Submission...")
    # run_experiment handles:
    # 1. Data loading/caching
    # 2. Model training with Early Stopping
    # 3. Generating submission.csv for the test set
    run_experiment(
        epochs=15,  # Limited epochs for fast baseline
        batch_size=1024,  # Large batch size for speed on GPU
        learning_rate=0.001,
        patience=5,
        load_cached_data=True,
    )

    # ---------------------------------------------------------
    # Phase 2: Validation Assessment
    # ---------------------------------------------------------
    print("\n[Phase 2] Validation Assessment...")

    # Load processed windowed validation data
    # X_val: (N_samples, Input_Dim), meta_val: List of (tripId, timestamp)
    X_val, y_val, meta_val = get_data(split="val", load_cached_data=True)

    # Load the raw aggregated validation dataframe from cache.
    # We need this to get the original WLS positions and Ground Truth coordinates
    # which are required to reconstruct predictions and calculate the metric.
    val_df_raw = pd.read_parquet(Config.CACHE_VAL_PATH)

    # Initialize the model architecture
    model = WindowedMLP(
        input_dim=Config.INPUT_DIM,
        hidden_layers=Config.HIDDEN_LAYERS,
        output_dim=Config.OUTPUT_DIM,
        dropout_rate=Config.DROPOUT_RATE,
    )

    # Load the best model weights saved during training
    if not os.path.exists(Config.MODEL_CHECKPOINT_PATH):
        raise FileNotFoundError("Model checkpoint not found. Training may have failed.")

    model.load_state_dict(
        torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=Config.DEVICE)
    )
    model.to(Config.DEVICE)
    model.eval()

    # Create DataLoader for validation inference
    val_dataset = GNSSDataset(X_val, mode="test")  # Mode 'test' returns only features
    val_loader = DataLoader(
        val_dataset,
        batch_size=2048,  # Larger batch size for faster inference (no gradients)
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Run Inference
    preds = []
    with torch.no_grad():
        for inputs in val_loader:
            inputs = inputs.to(Config.DEVICE)
            outputs = model(inputs)
            preds.append(outputs.cpu().numpy())

    preds = np.concatenate(preds, axis=0)

    # Create a DataFrame linking predictions to TripID and Timestamp
    trip_ids = [m[0] for m in meta_val]
    timestamps = [m[1] for m in meta_val]

    val_pred_df = pd.DataFrame(
        {
            Config.COL_TRIP_ID: trip_ids,
            Config.COL_UNIX_TIME: timestamps,
            "PredLatRes": preds[:, 0],  # Predicted Latitude Residual
            "PredLonRes": preds[:, 1],  # Predicted Longitude Residual
        }
    )

    # Merge predictions with raw data (WLS baseline and Ground Truth)
    # Inner join ensures we only evaluate on rows that were valid for the model
    val_analysis_df = pd.merge(
        val_pred_df,
        val_df_raw,
        on=[Config.COL_TRIP_ID, Config.COL_UNIX_TIME],
        how="inner",
    )

    # Reconstruct Final Predicted Coordinates
    # Prediction = Baseline (WLS) + Predicted Residual
    val_analysis_df["PredLat"] = (
        val_analysis_df[Config.FEAT_WLS_LAT] + val_analysis_df["PredLatRes"]
    )
    val_analysis_df["PredLon"] = (
        val_analysis_df[Config.FEAT_WLS_LON] + val_analysis_df["PredLonRes"]
    )

    # Calculate and Print Metric
    metric = calculate_competition_metric(val_analysis_df)
    print(f"Final Validation Metric: {metric}")

    # ---------------------------------------------------------
    # Phase 3: Failure Analysis
    # ---------------------------------------------------------
    print("\n[Phase 3] Failure Analysis...")
    perform_failure_analysis(val_analysis_df)

    print("\nPipeline execution completed.")


if __name__ == "__main__":
    main()
