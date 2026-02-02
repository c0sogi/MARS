import os
import sys
import numpy as np
import pandas as pd
import torch
import random

# Import provided library modules
import library.config as config
from library.data_loader import load_metadata
from library.feature_engineering import extract_features
from library.feature_selection import select_features
from library.model import train_model, predict_model, generate_submission


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # 1. Initialization
    set_seed(config.RANDOM_SEED)
    print("Initializing pipeline...")

    # 2. Data Loading & Feature Engineering
    # Load metadata
    print("Loading metadata...")
    train_meta = load_metadata("train")
    val_meta = load_metadata("val")
    test_meta = load_metadata("test")

    # Extract features (uses caching)
    print("Extracting features...")
    train_feats = extract_features(train_meta, "train", load_cached_data=True)
    val_feats = extract_features(val_meta, "val", load_cached_data=True)
    test_feats = extract_features(test_meta, "test", load_cached_data=True)

    # Merge targets into feature dataframes
    # The feature extraction returns stats + segment_id. We need to attach time_to_eruption.
    print("Merging targets...")
    train_merged = train_feats.merge(
        train_meta[[config.ID_COL, config.TARGET_COL]], on=config.ID_COL, how="left"
    )
    val_merged = val_feats.merge(
        val_meta[[config.ID_COL, config.TARGET_COL]], on=config.ID_COL, how="left"
    )

    # Prepare X and y for selection/training
    # Drop ID and Target from X
    X_train_full = train_merged.drop(columns=[config.ID_COL, config.TARGET_COL])
    y_train = train_merged[config.TARGET_COL]

    X_val_full = val_merged.drop(columns=[config.ID_COL, config.TARGET_COL])
    y_val = val_merged[config.TARGET_COL]

    # 3. Feature Selection
    # Cite solution_lesson_node_00017: Avoid aggressive wrapper-based feature selection (RFE)
    # for GBDTs. We skip selection and use the full feature set (~240 features).
    print("Skipping feature selection (using full feature set)...")

    X_train = X_train_full
    X_val = X_val_full

    # For test set, we need to keep ID_COL for submission generation later
    # We will create a clean version for prediction inside the loop or helper
    X_test_with_id = test_feats

    # 4. Model Training
    print("Training model...")
    model = train_model(X_train, y_train, X_val, y_val)

    # 5. Validation & Assessment
    print("Evaluating model...")
    # Predict on validation set
    val_preds = predict_model(model, X_val)

    # Calculate MAE
    mae = np.mean(np.abs(y_val - val_preds))
    print(f"Final Validation Metric: {mae}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate absolute errors
    abs_errors = np.abs(y_val - val_preds)

    # Create a dataframe for correlation analysis
    analysis_df = X_val.copy()
    analysis_df["abs_error"] = abs_errors

    # Calculate correlation between features and absolute error
    correlations = analysis_df.corr()["abs_error"].drop("abs_error")

    # Sort by absolute correlation
    top_correlations = correlations.abs().sort_values(ascending=False).head(10)

    print("Top 10 features correlated with model error:")
    for feature, corr_val in top_correlations.items():
        # Get the sign from the original correlation series
        sign = correlations[feature]
        print(f"{feature}: {sign:.4f}")

    # 6. Submission
    threshold = 2739761.26
    if mae < threshold:
        print(
            f"\nValidation metric ({mae}) meets threshold ({threshold}). Generating submission..."
        )
        generate_submission(model, X_test_with_id)
    else:
        print(
            f"\nValidation metric ({mae}) did not meet threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
