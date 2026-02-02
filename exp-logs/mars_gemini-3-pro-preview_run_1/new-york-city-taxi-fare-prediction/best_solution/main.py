import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import warnings

# Import provided library modules
from library import config
from library import utils
from library import data_processor

# from library import spatial_encoder  # Removed to simplify and revert to robust baseline
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
    # Cite solution_lesson_node_00002: Scale GBDT Capacity with Manifold-Aligned Spatial Features (Full Data)
    train_sample_frac = None  # Use full dataset
    print(f"Loading training data (sample_frac={train_sample_frac})...")
    train_df = dm.get_processed_data("train", sample_frac=train_sample_frac)
    print(f"Training data shape: {train_df.shape}")

    # Load Validation Data
    # Must use full validation set for accurate metric reporting
    print("Loading full validation data...")
    val_df = dm.get_processed_data("val", sample_frac=None)
    print(f"Validation data shape: {val_df.shape}")

    # Load Test Data
    print("Loading test data...")
    test_df = dm.get_processed_data("test")
    print(f"Test data shape: {test_df.shape}")

    # --- 3. Spatial Feature Engineering ---
    # Cite solution_lesson_node_00012: Removed Spatial Route Encoding to isolate validation fix
    # and return to the robust feature set of the "Current Best Solution".
    # The previous failure was due to validation outliers, but adding complexity now is risky.

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
    threshold = 3.8253698830539364

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
