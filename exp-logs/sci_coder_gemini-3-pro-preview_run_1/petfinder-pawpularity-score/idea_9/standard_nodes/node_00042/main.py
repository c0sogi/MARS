import os
import sys
import numpy as np
import pandas as pd
import torch

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, compute_rmse
from library.feature_extraction import FeatureExtractor
from library.stacking_engine import StackingEngine


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    print("Starting Runfile execution...")

    # 2. Feature Extraction
    # This will compute features if they don't exist in ./working
    print("\n=== Step 1: Feature Extraction ===")
    extractor = FeatureExtractor()
    extractor.run(load_cached_data=True)

    # 3. Stacking Ensemble Training
    print("\n=== Step 2: Stacking Ensemble Training ===")
    stacker = StackingEngine()

    # Train Level-0 Experts (returns OOF predictions and Test predictions)
    # This uses 5-Fold CV on the full dataset (Train + Val)
    oof_matrix, train_targets, test_matrix, test_ids = stacker.train_level0(
        load_cached_data=True
    )

    # Train Level-1 Meta-Learner on OOF predictions
    meta_learner = stacker.train_level1(oof_matrix, train_targets)

    # 4. Validation & Failure Analysis
    print("\n=== Step 3: Validation & Failure Analysis ===")

    # Predict on the full OOF matrix to get ensemble predictions
    ensemble_oof_preds = meta_learner.predict(oof_matrix)

    # Identify Validation Set portion
    # The 'train_all' dataset is constructed by concatenating Train then Val
    # We load the val metadata to know its size
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    n_val = len(val_df)
    n_total = len(train_targets)
    n_train_only = n_total - n_val

    # Slice the arrays to get validation set data
    # The OOF matrix preserves the order of the input data (Train followed by Val)
    val_preds = ensemble_oof_preds[n_train_only:]
    val_targets = train_targets[n_train_only:]

    # Compute Metric
    val_rmse = compute_rmse(val_targets, val_preds)
    print(f"Final Validation Metric: {val_rmse}")

    # Failure Analysis
    print("\n--- Failure Analysis (Correlation with Error Magnitude) ---")
    # Calculate absolute error
    errors = np.abs(val_preds - val_targets)

    # Get metadata features for validation set
    # Exclude non-feature columns
    feature_cols = [
        c for c in val_df.columns if c not in ["Id", "Pawpularity", "file_path"]
    ]
    val_features = val_df[feature_cols]

    # Compute correlations
    correlations = {}
    for col in feature_cols:
        # Handle potential constant columns to avoid warnings
        if val_features[col].std() == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(val_features[col], errors)[0, 1]
        correlations[col] = corr

    # Print sorted correlations
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, corr in sorted_corr:
        print(f"{feat}: {corr:.4f}")

    # 5. Submission
    print("\n=== Step 4: Submission Generation ===")
    THRESHOLD = 17.07053899184464

    if val_rmse < THRESHOLD:
        print(
            f"Validation metric ({val_rmse:.4f}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        stacker.generate_submission(meta_learner, test_matrix, test_ids)
    else:
        print(
            f"Validation metric ({val_rmse:.4f}) does NOT meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
