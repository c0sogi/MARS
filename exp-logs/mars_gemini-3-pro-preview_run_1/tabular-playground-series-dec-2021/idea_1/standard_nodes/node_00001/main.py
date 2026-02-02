import sys
import os
import pandas as pd
import numpy as np

# Add current directory to sys.path to ensure library imports work
sys.path.append(os.getcwd())

from library.utils import set_seed, save_submission
from library.data_manager import DataManager
from library.model_wrapper import XGBClassifierWrapper


def main():
    # --- 1. Setup & Configuration ---
    set_seed(42)
    print("Initializing baseline workflow...")

    # --- 2. Data Loading ---
    # Load full datasets. We do not subsample here to ensure we have the full
    # test set for submission and full validation set for accurate metrics.
    dm = DataManager()
    print("Loading datasets (using cache if available)...")
    dm.load_dataset(load_cached_data=True, sample_size=None)

    # --- 3. Preprocessing ---
    # Encode targets first to ensure the encoder sees all classes in the full dataset
    print("Encoding targets...")
    dm.encode_target()

    # Subsample training data for fast baseline execution as per requirements.
    # We use 500,000 samples which is sufficient for a strong baseline but fast to train.
    if dm.train_df is not None:
        print(f"Original train shape: {dm.train_df.shape}")
        dm.train_df = dm.train_df.sample(n=500000, random_state=42).reset_index(
            drop=True
        )
        print(f"Subsampled train shape: {dm.train_df.shape}")

    # --- 4. Prepare Data for Model ---
    print("Creating XGBoost DMatrices...")
    dtrain = dm.get_dmatrix("train")
    dval = dm.get_dmatrix("val")
    dtest = dm.get_dmatrix("test")

    # --- 5. Model Training ---
    print("Configuring model...")
    # Parameters optimized for A100 GPU acceleration
    params = {
        "objective": "multi:softmax",
        "num_class": 6,  # Mapped classes 0-5
        "tree_method": "hist",
        "device": "cuda",
        "eval_metric": ["merror"],
        "eta": 0.1,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbosity": 1,
        "seed": 42,
    }

    model_wrapper = XGBClassifierWrapper(
        params=params,
        num_boost_round=2000,  # Cap iterations for speed
        early_stopping_rounds=50,
        verbose_eval=100,
    )

    print("Starting training...")
    model_wrapper.train(dtrain, dval)

    # --- 6. Validation & Metrics ---
    print("Performing validation inference...")
    # Predict on full validation set
    val_preds = model_wrapper.predict(dval)

    # Get true encoded labels
    y_val = dm.val_df[dm.target_col].values

    # Calculate Accuracy
    accuracy = np.mean(val_preds == y_val)
    # Print exactly as requested
    print(f"Final Validation Metric: {accuracy}")

    # --- 7. Failure Analysis ---
    print("\n--- Failure Analysis ---")
    # Error mask: 1 if incorrect, 0 if correct
    errors = (val_preds != y_val).astype(int)

    # Calculate correlation between features and error magnitude
    val_features = dm.val_df[dm.feature_cols]
    correlations = {}

    print("Calculating feature correlations with error...")
    for col in val_features.columns:
        # Only process numeric columns (all are numeric in this dataset)
        if pd.api.types.is_numeric_dtype(val_features[col]):
            feat_values = val_features[col].values
            # Avoid correlation calculation if constant
            if np.std(feat_values) > 1e-9 and np.std(errors) > 1e-9:
                corr = np.corrcoef(feat_values, errors)[0, 1]
                correlations[col] = corr
            else:
                correlations[col] = 0.0

    # Sort by absolute correlation strength
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 features correlated with prediction error:")
    for name, corr in sorted_corrs[:10]:
        print(f"{name}: {corr:.6f}")

    # --- 8. Submission ---
    print("\nGenerating submission predictions...")
    test_preds_indices = model_wrapper.predict(dtest)

    # Inverse transform to get original Class IDs
    final_preds = dm.inverse_transform_target(test_preds_indices)

    # Get Test IDs
    test_ids = dm.get_test_ids()

    # Save submission
    submission_path = "./submission/submission.csv"
    print(f"Saving submission to {submission_path}...")
    save_submission(test_ids, final_preds, output_path=submission_path)

    print("Run complete.")


if __name__ == "__main__":
    main()
