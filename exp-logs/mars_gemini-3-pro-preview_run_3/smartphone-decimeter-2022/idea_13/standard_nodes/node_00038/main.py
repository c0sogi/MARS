import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

# Import from provided libraries
from library.config import INPUT_DIR, METADATA_DIR, SUBMISSION_DIR, haversine_distance
from library.data_loader import get_dataset
from library.model import train_model, predict


def main():
    # =========================================================================
    # 1. Data Loading
    # =========================================================================
    print("Loading datasets...")
    # Load datasets using the library function which handles caching and processing
    # We use load_cached_data=True to speed up if parquet files exist
    train_df = get_dataset(
        os.path.join(METADATA_DIR, "train_metadata.csv"),
        load_cached_data=True,
        split="train",
    )
    val_df = get_dataset(
        os.path.join(METADATA_DIR, "val_metadata.csv"),
        load_cached_data=True,
        split="val",
    )
    test_df = get_dataset(
        os.path.join(METADATA_DIR, "test_metadata.csv"),
        load_cached_data=True,
        split="test",
    )

    print(f"Train set shape: {train_df.shape}")
    print(f"Validation set shape: {val_df.shape}")
    print(f"Test set shape: {test_df.shape}")

    # =========================================================================
    # 2. Model Training
    # =========================================================================
    print("\nTraining models...")
    # Train the ensemble. train_model uses GroupKFold internally on the training set.
    # It returns the list of trained models for East and North components and the feature list.
    models_E, models_N, feature_cols = train_model(train_df)

    # =========================================================================
    # 3. Validation & Metric Calculation
    # =========================================================================
    print("\nRunning validation on hold-out set...")

    # Generate predictions for ENU residuals
    val_pred_E = predict(models_E, val_df, feature_cols)
    val_pred_N = predict(models_N, val_df, feature_cols)

    # Reconstruct Latitude/Longitude from predicted residuals
    # Formula:
    # Lat_pred = Lat_WLS + Delta_North / 111320
    # Lon_pred = Lon_WLS + Delta_East / (111320 * cos(Lat_WLS))
    wls_lat_rad = np.radians(val_df["wls_lat"])
    val_pred_lat = val_df["wls_lat"] + (val_pred_N / 111320.0)
    val_pred_lon = val_df["wls_lon"] + (val_pred_E / (111320.0 * np.cos(wls_lat_rad)))

    # Calculate Haversine Distance Error between Predicted and Ground Truth
    val_df["error_dist"] = haversine_distance(
        val_df["LatitudeDegrees"],
        val_df["LongitudeDegrees"],
        val_pred_lat,
        val_pred_lon,
    )

    # Compute Competition Metric: Mean of (50th + 95th percentile) per phone
    score_list = []
    # Group by tripId (which represents a unique phone run)
    for trip_id, group in val_df.groupby("tripId"):
        p50 = np.percentile(group["error_dist"], 50)
        p95 = np.percentile(group["error_dist"], 95)
        score_list.append((p50 + p95) / 2)

    final_metric = np.mean(score_list)
    print(f"Final Validation Metric: {final_metric}")

    # =========================================================================
    # 4. Failure Analysis
    # =========================================================================
    print("\nPerforming Failure Analysis...")

    # Analyze correlation between error magnitude and input features
    # This helps identify which conditions (e.g., low signal strength, high residuals) lead to errors
    analysis_df = val_df[feature_cols].copy()
    analysis_df["error_mag"] = val_df["error_dist"]

    # Compute correlations
    correlations = (
        analysis_df.corr()["error_mag"].drop("error_mag").sort_values(ascending=False)
    )

    print("Top 5 Features positively correlated with Error (High value -> High Error):")
    print(correlations.head(5))
    print(
        "\nTop 5 Features negatively correlated with Error (High value -> Low Error):"
    )
    print(correlations.tail(5))

    # =========================================================================
    # 5. Submission Generation
    # =========================================================================
    THRESHOLD = 4.29843913549805

    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric} is below threshold {THRESHOLD}. Generating submission..."
        )

        # Predict residuals for test set
        test_pred_E = predict(models_E, test_df, feature_cols)
        test_pred_N = predict(models_N, test_df, feature_cols)

        # Reconstruct Test Coordinates
        wls_lat_rad_test = np.radians(test_df["wls_lat"])
        test_pred_lat = test_df["wls_lat"] + (test_pred_N / 111320.0)
        test_pred_lon = test_df["wls_lon"] + (
            test_pred_E / (111320.0 * np.cos(wls_lat_rad_test))
        )

        # Create submission dataframe
        submission = pd.DataFrame(
            {
                "tripId": test_df["tripId"],
                "UnixTimeMillis": test_df.index,
                "LatitudeDegrees": test_pred_lat,
                "LongitudeDegrees": test_pred_lon,
            }
        )

        # Save
        os.makedirs(SUBMISSION_DIR, exist_ok=True)
        sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        submission.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

    else:
        print(
            f"\nMetric {final_metric} is NOT below threshold {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
