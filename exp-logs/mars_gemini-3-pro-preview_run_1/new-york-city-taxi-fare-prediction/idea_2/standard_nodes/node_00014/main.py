import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import warnings

# Import provided library modules
from library import config
from library import utils
from library import data_processor
from library import spatial_encoder
from library import model

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # --- 1. Setup ---
    print("--- Setting up environment ---")
    config.setup_directories()
    # Set random seeds for reproducibility
    np.random.seed(config.RANDOM_SEED)

    # --- 2. Data Loading & Preprocessing ---
    print("\n--- Data Loading & Processing ---")
    dm = data_processor.TaxiDataManager()

    # Load Training Data
    # Using full dataset for maximum performance
    # Cite solution_lesson_node_00002
    train_sample_frac = None
    print(f"Loading training data (sample_frac={train_sample_frac})...")
    # Force reload to apply new outlier filtering (< 500)
    train_df = dm.get_processed_data(
        "train", sample_frac=train_sample_frac, load_cached_data=False
    )
    print(f"Training data shape: {train_df.shape}")

    # Load Validation Data
    # Must use full validation set for accurate metric reporting
    print("Loading full validation data...")
    # Force reload to apply new outlier filtering (< 500)
    # Cite solution_lesson_node_00012
    val_df = dm.get_processed_data("val", sample_frac=None, load_cached_data=False)
    print(f"Validation data shape: {val_df.shape}")

    # Load Test Data
    print("Loading test data...")
    test_df = dm.get_processed_data("test")
    print(f"Test data shape: {test_df.shape}")

    # --- 3. Spatial Feature Engineering ---
    print("\n--- Spatial Target Encoding ---")
    encoder = spatial_encoder.SpatialRouteEncoder()

    # Fit clusters on training data
    print("Fitting spatial clusters...")
    encoder.fit_clusters(train_df)

    # Generate OOF features for training
    print("Generating OOF target encoding for training set...")
    # Force re-computation to match full dataset size and avoid stale cache
    train_oof = encoder.get_oof_target_encoding(train_df, load_cached_data=False)
    # Concatenate the new feature
    train_df = pd.concat([train_df, train_oof], axis=1)

    # Fit global map for inference
    print("Fitting global route map...")
    encoder.fit_global_map(train_df)

    # Apply global map to validation set
    print("Applying global target encoding to validation set...")
    val_enc = encoder.get_global_target_encoding(val_df, "val")
    val_df = pd.concat([val_df, val_enc], axis=1)

    # Apply global map to test set
    print("Applying global target encoding to test set...")
    test_enc = encoder.get_global_target_encoding(test_df, "test")
    test_df = pd.concat([test_df, test_enc], axis=1)

    # --- 4. Model Training ---
    print("\n--- Model Training ---")
    fare_model = model.FareModel()

    # Train the model
    # The model class handles feature selection (excluding key, target, etc.)
    fare_model.train(train_df, val_df)

    # --- 5. Evaluation ---
    print("\n--- Evaluation ---")
    # Predict on full validation set
    val_preds = fare_model.predict(val_df)

    # Calculate RMSE
    val_rmse = np.sqrt(mean_squared_error(val_df["fare_amount"], val_preds))
    print(f"Final Validation Metric: {val_rmse:.16f}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    val_df["prediction"] = val_preds
    val_df["abs_error"] = (val_df["fare_amount"] - val_df["prediction"]).abs()

    # Calculate correlations between absolute error and input features
    # Select numeric columns only
    numeric_cols = val_df.select_dtypes(include=[np.number]).columns
    correlations = {}

    skip_cols = ["abs_error", "prediction", "fare_amount", "key"]

    for col in numeric_cols:
        if col not in skip_cols:
            # Handle potential NaNs just in case, though processed data shouldn't have them
            if val_df[col].isna().any():
                continue
            corr = val_df["abs_error"].corr(val_df[col])
            correlations[col] = corr

    # Sort and print top correlations
    print("Top feature correlations with error magnitude:")
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, val in sorted_corr[:5]:
        print(f"  {name}: {val:.4f}")

    # --- 6. Submission ---
    print("\n--- Submission Generation ---")
    threshold = 3.8031316464284455

    if val_rmse < threshold:
        print(
            f"Validation RMSE ({val_rmse:.4f}) < Threshold ({threshold:.4f}). Generating submission..."
        )

        test_preds = fare_model.predict(test_df)

        submission = pd.DataFrame({"key": test_df["key"], "fare_amount": test_preds})

        # Save submission
        submission.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")

    else:
        print(
            f"Validation RMSE ({val_rmse:.4f}) >= Threshold ({threshold:.4f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
