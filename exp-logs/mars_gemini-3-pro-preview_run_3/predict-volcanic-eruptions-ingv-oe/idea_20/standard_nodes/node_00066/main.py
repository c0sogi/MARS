import os
import gc
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

# Import from provided library
from library.config import Config
from library.feature_engineering import generate_features
from library.model import VolcanoLGBM


def main():
    # 1. Configuration and Setup
    # Ensure reproducibility
    np.random.seed(Config.SEED)

    print("Initializing Volcano Eruption Prediction Pipeline...")

    # 2. Feature Generation / Loading
    # We use the generate_features function which handles caching and parallel processing
    print("\n--- Loading Data ---")

    # Load Train Data
    train_df = generate_features(
        Config.TRAIN_METADATA_PATH, Config.TRAIN_FEATURES_PATH, load_cached_data=True
    )

    # Load Validation Data (Hold-out set)
    val_df = generate_features(
        Config.VAL_METADATA_PATH, Config.VAL_FEATURES_PATH, load_cached_data=True
    )

    # Load Test Data
    test_df = generate_features(
        Config.TEST_METADATA_PATH, Config.TEST_FEATURES_PATH, load_cached_data=True
    )

    # 3. Data Preparation
    print("\n--- Preparing Feature Matrices ---")
    # Identify feature columns (exclude metadata columns)
    ignore_cols = ["segment_id", "time_to_eruption"]
    feature_cols = [c for c in train_df.columns if c not in ignore_cols]

    print(f"Number of features: {len(feature_cols)}")

    # Create Numpy arrays for modeling
    X_train = train_df[feature_cols].values
    y_train = train_df["time_to_eruption"].values

    X_val = val_df[feature_cols].values
    y_val = val_df["time_to_eruption"].values

    X_test = test_df[feature_cols].values
    test_ids = test_df["segment_id"].values

    # Clean up DataFrames to save memory
    del train_df, val_df, test_df
    gc.collect()

    # 4. Model Training
    print("\n--- Training Model ---")
    model_wrapper = VolcanoLGBM()

    # Train on Train set, Validate on Hold-out Val set
    # The wrapper handles early stopping internally using the provided sets
    model_wrapper.train(X_train, y_train, X_val, y_val)

    # 5. Validation & Metrics
    print("\n--- Validating ---")
    # Predict on validation set
    val_preds = model_wrapper.predict(X_val)

    # Compute Metric (MAE)
    val_mae = mean_absolute_error(y_val, val_preds)

    # Print the required metric string
    print(f"Final Validation Metric: {val_mae}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate absolute errors
    errors = np.abs(y_val - val_preds)

    # Compute correlation between each feature and the error magnitude
    # This helps identify which features are associated with high prediction errors
    feature_corrs = []

    # Normalize errors for correlation calculation
    err_mean = np.mean(errors)
    err_std = np.std(errors)

    if err_std > 0:
        # Vectorized correlation calculation for speed
        # Formula: E[(X - mu_x)(Y - mu_y)] / (sigma_x * sigma_y)

        # Center and standardize features
        X_val_mean = np.mean(X_val, axis=0)
        X_val_centered = X_val - X_val_mean
        X_val_std = np.std(X_val, axis=0)

        # Avoid division by zero for constant features
        valid_feats_mask = X_val_std > 1e-9

        # Center errors
        err_centered = (errors - err_mean).reshape(-1, 1)

        # Compute covariance
        # (N, F)^T dot (N, 1) -> (F, 1)
        covariance = np.dot(X_val_centered.T, err_centered).flatten() / len(errors)

        # Compute correlation
        correlations = np.zeros(len(feature_cols))
        # Only compute for valid features
        if np.any(valid_feats_mask):
            correlations[valid_feats_mask] = covariance[valid_feats_mask] / (
                X_val_std[valid_feats_mask] * err_std
            )

        # Map back to feature names
        for i, name in enumerate(feature_cols):
            feature_corrs.append((name, correlations[i]))

    else:
        print("Error standard deviation is 0. Cannot compute correlation.")

    # Sort by absolute correlation strength
    feature_corrs.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 Features associated with Error (Correlation):")
    for name, corr in feature_corrs[:10]:
        print(f"{name}: {corr:.4f}")

    # 7. Submission
    # Threshold defined in the task
    THRESHOLD = 2617304.0647319085

    if val_mae < THRESHOLD:
        print(
            f"\nValidation Metric ({val_mae}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Predict on Test set
        test_preds = model_wrapper.predict(X_test)

        # Create submission DataFrame
        submission_df = pd.DataFrame(
            {"segment_id": test_ids, "time_to_eruption": test_preds}
        )

        # Ensure output directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        # Save submission
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation Metric ({val_mae}) does NOT meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
