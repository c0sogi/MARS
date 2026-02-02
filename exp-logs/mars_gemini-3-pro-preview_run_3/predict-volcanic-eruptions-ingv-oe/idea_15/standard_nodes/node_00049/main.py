import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

# Import from the provided library files
from library import config
from library.feature_engineering import process_dataset
from library.model_trainer import train_lgbm


def main():
    # Set seeds for reproducibility
    np.random.seed(config.SEED)

    # ==========================================
    # 1. Data Loading & Feature Engineering
    # ==========================================
    print("Step 1: Loading and processing data...")

    # Load Train and Validation data
    # We use the full datasets (debug=False) to ensure we meet the metric threshold.
    # The dataset size (~4000 samples) is small enough for rapid LightGBM training.
    # We enable caching to speed up subsequent runs.
    train_df = process_dataset("train", load_cached_data=True, debug=False)
    val_df = process_dataset("val", load_cached_data=True, debug=False)

    # Identify feature columns (exclude metadata and target)
    exclude_cols = ["segment_id", "time_to_eruption"]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    print(f"Training on {len(train_df)} samples with {len(feature_cols)} features.")
    print(f"Validating on {len(val_df)} samples.")

    # Prepare matrices
    X_train = train_df[feature_cols]
    y_train = train_df["time_to_eruption"]

    X_val = val_df[feature_cols]
    y_val = val_df["time_to_eruption"]

    # ==========================================
    # 2. Model Training
    # ==========================================
    print("Step 2: Training LightGBM model...")

    # Use parameters from config
    params = config.LGBM_PARAMS.copy()
    # Explicitly silence LightGBM
    params["verbose"] = -1

    # Train the model
    # We use the validation set for early stopping to prevent overfitting
    model = train_lgbm(X_train, y_train, X_val, y_val, params)

    # ==========================================
    # 3. Validation Assessment
    # ==========================================
    print("Step 3: Evaluating model...")

    # Predict on validation set
    # LightGBM inference is highly optimized and automatically handles CPU/GPU usage based on build
    y_pred_val = model.predict(X_val)

    # Calculate MAE
    val_mae = mean_absolute_error(y_val, y_pred_val)

    # Print the required metric in the specified format
    print(f"Final Validation Metric: {val_mae}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("Step 4: Performing Failure Analysis...")

    # Calculate absolute errors
    errors = np.abs(y_val - y_pred_val)

    # Calculate correlation between features and error magnitude
    correlations = []
    # Convert to float to avoid any object type issues
    X_val_float = X_val.astype(float)

    for col in X_val_float.columns:
        # Skip constant columns to avoid division by zero in correlation
        if X_val_float[col].std() == 0:
            continue

        # Calculate correlation coefficient
        corr = np.corrcoef(X_val_float[col], errors)[0, 1]

        if not np.isnan(corr):
            correlations.append((col, corr))

    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 features correlated with error magnitude:")
    for feat, corr in correlations[:10]:
        print(f"{feat}: {corr:.6f}")

    # ==========================================
    # 5. Submission Generation
    # ==========================================
    print("Step 5: Checking submission criteria...")

    TARGET_METRIC = 2739761.2592384242

    if val_mae < TARGET_METRIC:
        print(
            f"Validation MAE ({val_mae}) is better than target ({TARGET_METRIC}). Generating submission..."
        )

        # Load Test data
        test_df = process_dataset("test", load_cached_data=True, debug=False)

        # Ensure test features match training features
        X_test = test_df[feature_cols]

        # Generate predictions
        test_preds = model.predict(X_test)

        # Create submission DataFrame
        submission_df = pd.DataFrame(
            {"segment_id": test_df["segment_id"], "time_to_eruption": test_preds}
        )

        # Save submission
        save_path = config.SUBMISSION_PATH
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        submission_df.to_csv(save_path, index=False)

        print(f"Submission saved to {save_path}")

    else:
        print(
            f"Validation MAE ({val_mae}) did not meet target ({TARGET_METRIC}). Submission skipped."
        )


if __name__ == "__main__":
    main()
