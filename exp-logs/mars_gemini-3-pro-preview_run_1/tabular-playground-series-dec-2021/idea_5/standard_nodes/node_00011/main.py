import os
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import accuracy_score

# Import provided library components
from library import config
from library.data_processor import engineer_features
from library.ensemble_trainer import train_ensemble
from library.inference_utils import soft_vote_predict, export_submission


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # --- 1. Setup ---
    warnings.filterwarnings("ignore")
    set_seed(config.SEED)

    # --- 2. Data Loading ---
    # We load from metadata to respect the hold-out validation requirement strictly.
    # Training on metadata/train.csv (80%) and validating on metadata/val.csv (20%).
    # This prevents data leakage that would occur if we trained on the full input/train.csv.
    print("Loading datasets...")
    train_df = pd.read_csv(config.METADATA_TRAIN_PATH)
    val_df = pd.read_csv(config.METADATA_VAL_PATH)
    # Load test from raw input to ensure submission ID consistency
    test_df = pd.read_csv(config.RAW_TEST_PATH)

    # --- 3. Feature Engineering ---
    print("Applying feature engineering...")
    # Apply the physics-informed geometric transformations
    train_df = engineer_features(train_df)
    val_df = engineer_features(val_df)
    test_df = engineer_features(test_df)

    # --- 4. Preprocessing ---
    target_col = "Cover_Type"
    id_col = "Id"

    # Prepare Training Data
    y_train = train_df[target_col]
    X_train = train_df.drop(columns=[target_col])
    if id_col in X_train.columns:
        X_train = X_train.drop(columns=[id_col])

    # Prepare Validation Data
    y_val = val_df[target_col]
    X_val = val_df.drop(columns=[target_col])
    if id_col in X_val.columns:
        X_val = X_val.drop(columns=[id_col])

    # Prepare Test Data
    test_ids = test_df[id_col]
    X_test = test_df.drop(columns=[id_col])

    # Ensure column alignment across all sets
    X_val = X_val[X_train.columns]
    X_test = X_test[X_train.columns]

    # --- 5. Training ---
    # Train the homogeneous ensemble on the training subset using Stratified K-Fold
    # The ensemble_trainer handles the split and training loop internally.
    # XGBoost is configured for GPU acceleration in config.XGB_PARAMS.
    models, oof_scores = train_ensemble(
        X_train, y_train, config.XGB_PARAMS, n_folds=config.N_FOLDS
    )

    # --- 6. Validation ---
    print("Performing inference on hold-out validation set...")
    # Aggregate predictions from all 5 models using soft voting
    val_preds = soft_vote_predict(models, X_val)
    final_val_acc = accuracy_score(y_val, val_preds)

    # REQUIRED OUTPUT: Print the final validation metric
    print(f"Final Validation Metric: {final_val_acc}")

    # --- 7. Failure Analysis ---
    print("Running failure analysis...")
    # Create error mask: 1 if prediction is wrong, 0 if correct
    error_mask = (val_preds != y_val).astype(int)

    correlations = []
    # Calculate correlation between each feature and the error mask
    for col in X_val.columns:
        # Ensure column is numeric before correlation
        if pd.api.types.is_numeric_dtype(X_val[col]):
            try:
                # Compute Point-Biserial correlation (simplified as Pearson here)
                corr = np.corrcoef(X_val[col], error_mask)[0, 1]
                if not np.isnan(corr):
                    correlations.append((col, corr))
            except Exception:
                continue

    # Sort by absolute correlation strength (magnitude indicates relationship strength)
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Prediction Error:")
    for feat, corr in correlations[:5]:
        print(f"{feat}: {corr:.6f}")

    # --- 8. Submission ---
    # Only submit if the model meets the performance requirement
    threshold = 0.9614708333333334
    if final_val_acc > threshold:
        print(
            f"Metric ({final_val_acc}) > Threshold ({threshold}). Generating submission..."
        )
        # Generate predictions for the test set
        test_preds = soft_vote_predict(models, X_test)
        # Save to CSV
        export_submission(test_ids, test_preds, config.SUBMISSION_PATH)
    else:
        print(
            f"Metric ({final_val_acc}) <= Threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
