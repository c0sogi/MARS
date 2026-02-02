import os
import sys
import warnings
import pandas as pd
import numpy as np
import torch

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, compute_rmse
from library.workflow import StackingManager

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup and Reproducibility
    seed_everything(Config.SEED)

    # 2. Initialize Stacking Manager
    # We use the full dataset (subset_size=None) as the A100 can handle 7k images
    # and feature extraction very quickly (within minutes).
    # load_cached_data=True allows skipping feature extraction if files exist.
    manager = StackingManager(
        subset_size=None,
        load_cached_data=True,
        n_folds=Config.N_FOLDS,
        working_dir=Config.WORKING_DIR,
    )

    # 3. Train Level 0 Experts (Swin, ConvNeXt, CLIP)
    # This step extracts features (if not cached) and trains RidgeCV models.
    manager.train_level_0()

    # 4. Train Level 1 Meta-Learner
    # This step trains the stacking regressor on OOF predictions.
    manager.train_level_1()

    # 5. Validation Assessment & Failure Analysis
    print("\n=== Validation Assessment & Failure Analysis ===")

    # Reconstruct final validation predictions to calculate metrics and analysis
    # We stack the validation predictions from all experts
    sorted_keys = sorted(manager.model_keys)
    X_val_L1 = np.column_stack([manager.L0_val_preds[k] for k in sorted_keys])

    # Predict using the trained meta-learner
    val_preds = manager.meta_learner.predict(X_val_L1)
    val_targets = manager.val_targets.ravel()

    # Calculate Final Metric
    final_rmse = compute_rmse(val_targets, val_preds)
    print(f"Final Validation Metric: {final_rmse}")

    # Failure Analysis: Correlation between Error Magnitude and Features
    # Load validation metadata
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Calculate absolute errors
    errors = np.abs(val_targets - val_preds)

    # Create a DataFrame for analysis
    analysis_df = val_df[Config.METADATA_COLS].copy()
    analysis_df["Error_Magnitude"] = errors

    # Compute correlations
    print("\nFailure Analysis (Correlation with Error Magnitude):")
    correlations = analysis_df.corr()["Error_Magnitude"].drop("Error_Magnitude")
    print(correlations.sort_values(ascending=False).to_string())

    # 6. Submission Generation
    # Threshold defined in the task description
    THRESHOLD = 17.459202675242267

    if final_rmse < THRESHOLD:
        print(
            f"\nValidation RMSE ({final_rmse}) is lower than threshold ({THRESHOLD})."
        )
        print("Generating submission file...")
        manager.predict()
    else:
        print(
            f"\nValidation RMSE ({final_rmse}) is NOT lower than threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
