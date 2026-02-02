import sys
import os
import numpy as np
import pandas as pd
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import functions and classes from the provided library files
from library.config import SUBMISSION_PATH, SEED
from library.dataset import get_features_and_targets
from library.model import LGBMRegressorWrapper, generate_submission
from library.utils import haversine_distance, compute_percentile_error


def main():
    # Set fixed random seed for reproducibility
    np.random.seed(SEED)

    print("========================================")
    print(" STARTING PIPELINE")
    print("========================================")

    # ---------------------------------------------------------
    # 1. Data Loading
    # ---------------------------------------------------------
    print("\n[1/5] Loading Datasets...")

    # Load Training Data
    # load_cached_data=True attempts to load pre-computed parquet files from ./working/idea_2/
    # If not found, it triggers the feature engineering pipeline.
    X_train, y_train, _ = get_features_and_targets(split="train", load_cached_data=True)
    print(f"  Training Features Shape: {X_train.shape}")

    # Load Validation Data
    # We need the full dataframe (df_val) to access baseline WLS positions and Ground Truth
    X_val, y_val, df_val = get_features_and_targets(split="val", load_cached_data=True)
    print(f"  Validation Features Shape: {X_val.shape}")

    # ---------------------------------------------------------
    # 2. Model Training
    # ---------------------------------------------------------
    print("\n[2/5] Training Models...")

    # Initialize the LightGBM wrapper
    # We set n_estimators to 2000, but early stopping (configured in model.py) will prevent overfitting
    model = LGBMRegressorWrapper(n_estimators=2000)

    # Train the models (Latitude and Longitude regressors are trained independently)
    model.train(X_train, y_train, X_val, y_val)

    # ---------------------------------------------------------
    # 3. Validation Assessment
    # ---------------------------------------------------------
    print("\n[3/5] Validating...")

    # Generate predictions on the validation set (Predicted Errors)
    pred_lat_err, pred_lon_err = model.predict(X_val)

    # Reconstruct final predicted positions
    # Final Position = Baseline WLS Position + Predicted Error
    val_pred_lat = df_val["lat_wls"] + pred_lat_err
    val_pred_lon = df_val["lon_wls"] + pred_lon_err

    # Calculate Haversine distance error (in meters) between Predicted and Ground Truth
    distances = haversine_distance(
        df_val["LatitudeDegrees"].values,
        df_val["LongitudeDegrees"].values,
        val_pred_lat.values,
        val_pred_lon.values,
    )

    # Add calculated distances to the dataframe to facilitate grouping
    df_val["dist_error"] = distances

    # Compute the competition metric: Mean of (Mean of (50th + 95th percentile)) per phone
    trip_scores = []
    # Group by tripId (which represents a unique phone-drive combination)
    for trip_id, group in df_val.groupby("tripId"):
        p50, p95 = compute_percentile_error(group["dist_error"].values)
        score = (p50 + p95) / 2
        trip_scores.append(score)

    final_metric = np.mean(trip_scores)
    print(f"Final Validation Metric: {final_metric}")

    # ---------------------------------------------------------
    # 4. Failure Analysis
    # ---------------------------------------------------------
    print("\n[4/5] Performing Failure Analysis...")

    # Analyze which features correlate most with high position errors
    analysis_df = X_val.copy()
    analysis_df["Error_Magnitude"] = distances

    # Compute Pearson correlation
    correlations = analysis_df.corrwith(analysis_df["Error_Magnitude"]).sort_values(
        ascending=False
    )

    print("Correlation between Input Features and Error Magnitude:")
    print(correlations)

    # ---------------------------------------------------------
    # 5. Inference & Submission
    # ---------------------------------------------------------
    print("\n[5/5] Generating Submission...")

    # Load Test Data
    # y_test is None, df_test contains metadata and baseline WLS for the test set
    X_test, _, df_test = get_features_and_targets(split="test", load_cached_data=True)
    print(f"  Test Features Shape: {X_test.shape}")

    # Predict corrections for the test set
    test_lat_err, test_lon_err = model.predict(X_test)

    # Apply corrections to test baseline and save to CSV
    generate_submission(df_test, test_lat_err, test_lon_err, SUBMISSION_PATH)

    print("\nPipeline Complete.")


if __name__ == "__main__":
    main()
