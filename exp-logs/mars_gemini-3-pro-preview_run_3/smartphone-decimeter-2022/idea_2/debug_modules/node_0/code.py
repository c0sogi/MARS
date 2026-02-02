import os
import pandas as pd
import numpy as np
import warnings

# Import from the provided library
from library.features import generate_dataset
from library.model import LGBMRegressorWrapper, generate_submission
from library.config import (
    AGG_FEATURES,
    TARGETS,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    WORKING_DIR,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("Starting End-to-End Pipeline Demonstration...")

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Create Subsampled Metadata for Quick Execution
    # ---------------------------------------------------------
    print("\n[1] Subsampling Metadata for Demo...")

    def create_subset_metadata(source_path, dest_path, n_trips=2):
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source metadata not found: {source_path}")

        df = pd.read_csv(source_path)
        # Select first n_trips
        selected_trips = df["tripId"].unique()[:n_trips]
        subset_df = df[df["tripId"].isin(selected_trips)].copy()

        subset_df.to_csv(dest_path, index=False)
        print(
            f"    Created subset {dest_path} with {len(subset_df)} rows ({n_trips} trips)."
        )
        return len(subset_df)

    # Define temporary paths
    temp_train_meta = os.path.join(WORKING_DIR, "demo_train_meta.csv")
    temp_val_meta = os.path.join(WORKING_DIR, "demo_val_meta.csv")
    temp_test_meta = os.path.join(WORKING_DIR, "demo_test_meta.csv")

    # Create subsets
    n_train_rows = create_subset_metadata(
        TRAIN_METADATA_PATH, temp_train_meta, n_trips=2
    )
    n_val_rows = create_subset_metadata(VAL_METADATA_PATH, temp_val_meta, n_trips=1)
    n_test_rows = create_subset_metadata(TEST_METADATA_PATH, temp_test_meta, n_trips=1)

    # ---------------------------------------------------------
    # 2. Feature Generation
    # ---------------------------------------------------------
    print("\n[2] Generating Features from Raw Sensor Data...")

    # We use generate_dataset directly to specify our temp metadata paths
    # We set load_cached_data=False to force processing for this demo

    print("    Processing Training Data...")
    df_train = generate_dataset(
        temp_train_meta, load_cached_data=False, split_name="demo_train"
    )

    print("    Processing Validation Data...")
    df_val = generate_dataset(
        temp_val_meta, load_cached_data=False, split_name="demo_val"
    )

    print("    Processing Test Data...")
    df_test = generate_dataset(
        temp_test_meta, load_cached_data=False, split_name="demo_test"
    )

    # Verify Data Generation
    assert not df_train.empty, "Training dataframe is empty!"
    assert not df_val.empty, "Validation dataframe is empty!"
    assert not df_test.empty, "Test dataframe is empty!"

    print(f"    Generated Train Shape: {df_train.shape}")
    print(f"    Generated Val Shape:   {df_val.shape}")
    print(f"    Generated Test Shape:  {df_test.shape}")

    # ---------------------------------------------------------
    # 3. Data Preparation for Modeling
    # ---------------------------------------------------------
    print("\n[3] Preparing Feature Matrices and Targets...")

    # Identify available features (intersection of config features and generated columns)
    # Some features might be missing if IMU data was missing for a trip, handling gracefully
    feature_cols = [f for f in AGG_FEATURES if f in df_train.columns]
    print(f"    Using {len(feature_cols)} features: {feature_cols}")

    # Prepare Train
    X_train = df_train[feature_cols]
    y_train = df_train[TARGETS]

    # Prepare Val
    X_val = df_val[feature_cols]
    y_val = df_val[TARGETS]

    # Prepare Test
    X_test = df_test[feature_cols]

    # Verify Targets exist
    assert "lat_error" in y_train.columns and "lon_error" in y_train.columns
    assert not y_train.isnull().any().any(), "NaNs found in training targets"

    # ---------------------------------------------------------
    # 4. Model Training
    # ---------------------------------------------------------
    print("\n[4] Training LightGBM Models...")

    # Initialize model with low estimators for speed
    model = LGBMRegressorWrapper(
        n_estimators=10,
        learning_rate=0.1,
        num_leaves=10,
        min_child_samples=5,  # Reduced because data is small
    )

    # Train
    model.train(X_train, y_train, X_val, y_val)

    print("    Training complete.")

    # ---------------------------------------------------------
    # 5. Inference
    # ---------------------------------------------------------
    print("\n[5] Running Inference on Test Set...")

    lat_preds, lon_preds = model.predict(X_test)

    # Basic validation of predictions
    assert len(lat_preds) == len(df_test)
    assert len(lon_preds) == len(df_test)
    assert np.isfinite(lat_preds).all(), "NaN/Inf in latitude predictions"
    assert np.isfinite(lon_preds).all(), "NaN/Inf in longitude predictions"

    print(
        f"    Predictions generated. Lat mean: {lat_preds.mean():.4f}, Lon mean: {lon_preds.mean():.4f}"
    )

    # ---------------------------------------------------------
    # 6. Submission Generation
    # ---------------------------------------------------------
    print("\n[6] Generating Submission File...")

    submission_output_path = os.path.join(WORKING_DIR, "demo_submission.csv")

    generate_submission(df_test, lat_preds, lon_preds, submission_output_path)

    # Verify output
    if os.path.exists(submission_output_path):
        sub_df = pd.read_csv(submission_output_path)
        print(f"    Submission file verified. Shape: {sub_df.shape}")
        print(f"    Columns: {list(sub_df.columns)}")

        expected_cols = [
            "tripId",
            "UnixTimeMillis",
            "LatitudeDegrees",
            "LongitudeDegrees",
        ]
        assert list(sub_df.columns) == expected_cols, "Submission columns mismatch"
        assert len(sub_df) == len(df_test), "Submission row count mismatch"
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\nDemonstration Completed Successfully!")


if __name__ == "__main__":
    main()
