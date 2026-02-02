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
    # Phase 1: Training
    # ---------------------------------------------------------
    print("\n[Phase 1] Training Model...")
    # run_experiment handles data loading/caching and model training
    trained_model = run_experiment(
        epochs=30,
        batch_size=1024,
        learning_rate=0.001,
        patience=5,
        load_cached_data=False,  # Force reload to incorporate new delta features
    )

    # ---------------------------------------------------------
    # Phase 2: Validation Assessment
    # ---------------------------------------------------------
    print("\n[Phase 2] Validation Assessment...")

    # Load processed windowed validation data
    X_val, y_val, meta_val = get_data(split="val", load_cached_data=True)

    # Load the raw aggregated validation dataframe from cache
    val_df_raw = pd.read_parquet(Config.CACHE_VAL_PATH)

    # Use the trained model directly
    trained_model.eval()

    # Create DataLoader for validation inference
    val_dataset = GNSSDataset(X_val, mode="test")
    val_loader = DataLoader(
        val_dataset,
        batch_size=2048,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Run Inference
    preds = []
    with torch.no_grad():
        for inputs in val_loader:
            inputs = inputs.to(Config.DEVICE)
            outputs = trained_model(inputs)
            preds.append(outputs.cpu().numpy())

    preds = np.concatenate(preds, axis=0)

    # Create a DataFrame linking predictions to TripID and Timestamp
    trip_ids = [m[0] for m in meta_val]
    timestamps = [m[1] for m in meta_val]

    val_pred_df = pd.DataFrame(
        {
            Config.COL_TRIP_ID: trip_ids,
            Config.COL_UNIX_TIME: timestamps,
            "PredLatRes": preds[:, 0] / Config.TARGET_SCALE_FACTOR,
            "PredLonRes": preds[:, 1] / Config.TARGET_SCALE_FACTOR,
        }
    )

    # Merge predictions with raw data
    val_analysis_df = pd.merge(
        val_pred_df,
        val_df_raw,
        on=[Config.COL_TRIP_ID, Config.COL_UNIX_TIME],
        how="inner",
    )

    # Reconstruct Final Predicted Coordinates
    val_analysis_df["PredLat"] = (
        val_analysis_df[Config.FEAT_WLS_LAT] + val_analysis_df["PredLatRes"]
    )
    val_analysis_df["PredLon"] = (
        val_analysis_df[Config.FEAT_WLS_LON] + val_analysis_df["PredLonRes"]
    )

    # Calculate Metric
    metric = calculate_competition_metric(val_analysis_df)
    print(f"Final Validation Metric: {metric}")

    # Failure Analysis
    perform_failure_analysis(val_analysis_df)

    # ---------------------------------------------------------
    # Phase 3: Conditional Submission Generation
    # ---------------------------------------------------------
    THRESHOLD = 4.324936265133881
    if metric < THRESHOLD:
        print(
            f"\n[Phase 3] Metric {metric:.4f} < {THRESHOLD:.4f}. Generating Submission..."
        )

        # Import here to avoid circular dependency issues if any, though trainer imports it
        from library.model import generate_submission

        # Load test data
        X_test, meta_test, df_test_original = get_data(
            split="test", load_cached_data=False
        )

        test_dataset = GNSSDataset(X_test, mode="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=2048,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        generate_submission(
            model=trained_model,
            test_loader=test_loader,
            meta_list=meta_test,
            df_test_original=df_test_original,
            submission_path=Config.SUBMISSION_FILE_PATH,
            device=Config.DEVICE,
        )
    else:
        print(
            f"\n[Phase 3] Metric {metric:.4f} >= {THRESHOLD:.4f}. Skipping Submission."
        )

    print("\nPipeline execution completed.")


if __name__ == "__main__":
    main()
