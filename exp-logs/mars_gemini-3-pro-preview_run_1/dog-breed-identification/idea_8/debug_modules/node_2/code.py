import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library modules
from library.config import Config
from library.utils import seed_everything
from library.dataset import _process_data
from library.training import run_regime_a, run_regime_b
from library.stacking import Stacker


def main():
    print("Starting Demonstration Script...")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Override Config to use a specific demo directory.
    # This ensures we don't overwrite existing work and keeps the demo self-contained.
    Config.WORK_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = "./working/demo_execution"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure the working directory exists
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    # Set random seed for reproducibility
    seed_everything(Config.SEED)
    print(f"Configuration set. Working directory: {Config.WORK_DIR}")

    # -------------------------------------------------------------------------
    # 2. Data Pipeline Verification
    # -------------------------------------------------------------------------
    print("\n=== Step 1: Data Processing ===")
    # We call _process_data with load_cached_data=False to force the processing
    # logic to run (resize, crop, save to .npy).
    train_imgs, train_targets, train_ids, test_imgs, test_ids, label_map = (
        _process_data(load_cached_data=False)
    )

    # Verification
    print(f"Processed Train Images Shape: {train_imgs.shape}")
    print(f"Processed Test Images Shape: {test_imgs.shape}")
    print(f"Number of Classes: {len(label_map)}")

    # Assertions to ensure data integrity
    assert train_imgs.ndim == 4, "Train images should be 4D (N, H, W, C)"
    assert train_imgs.shape[0] == len(
        train_targets
    ), "Mismatch between images and targets"
    assert train_imgs.shape[0] == len(train_ids), "Mismatch between images and IDs"
    assert (
        len(label_map) == Config.NUM_CLASSES
    ), f"Expected {Config.NUM_CLASSES} classes, found {len(label_map)}"

    print("Data processing verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Training Regime A (ConvNeXt)
    # -------------------------------------------------------------------------
    print("\n=== Step 2: Training Regime A (ConvNeXt) - Fold 0 [Debug Mode] ===")
    # Run the training loop for the first fold in debug mode.
    # Debug mode limits the number of epochs (to 1 per phase) and batches (to 10),
    # ensuring this completes in seconds/minutes rather than hours.
    run_regime_a(fold_idx=0, debug=True)

    # Verify that the model artifact was saved
    model_a_path = os.path.join(Config.WORK_DIR, "convnext_base_fold_0.pth")
    if not os.path.exists(model_a_path):
        raise FileNotFoundError(
            f"Training Regime A failed to save model at {model_a_path}"
        )

    print(f"Model A artifact verified at: {model_a_path}")

    # -------------------------------------------------------------------------
    # 4. Training Regime B (Swin Transformer)
    # -------------------------------------------------------------------------
    print("\n=== Step 3: Training Regime B (Swin) - Fold 0 [Debug Mode] ===")
    # Run the training loop for the second regime.
    # This exercises the Mixup/CutMix logic and different optimizer schedules.
    run_regime_b(fold_idx=0, debug=True)

    # Verify that the model artifact was saved
    model_b_path = os.path.join(Config.WORK_DIR, "swin_base_fold_0.pth")
    if not os.path.exists(model_b_path):
        raise FileNotFoundError(
            f"Training Regime B failed to save model at {model_b_path}"
        )

    print(f"Model B artifact verified at: {model_b_path}")

    # -------------------------------------------------------------------------
    # 5. Stacking Ensemble
    # -------------------------------------------------------------------------
    print("\n=== Step 4: Stacking Ensemble [Debug Mode] ===")
    # Initialize Stacker in debug mode (sets n_folds=1)
    stacker = Stacker(debug=True)

    # Generate OOF and Test predictions using the models trained above.
    # We disable cache loading to force inference to run.
    print("Generating predictions from expert models...")
    stacking_data = stacker.get_data(load_cached_data=False)

    # Verify the structure of the returned data dictionary
    required_keys = [
        "oof_preds_a",
        "oof_preds_b",
        "oof_targets",
        "test_preds_a",
        "test_preds_b",
        "test_ids",
    ]
    for key in required_keys:
        if key not in stacking_data:
            raise KeyError(f"Stacking data missing required key: {key}")

    print(f"OOF Predictions Shape: {stacking_data['oof_preds_a'].shape}")
    print(f"Test Predictions Shape: {stacking_data['test_preds_a'].shape}")

    # Train the Meta-Learner (Logistic Regression) on OOF data
    print("Training Meta-Learner...")
    meta_model = stacker.train_meta_learner(stacking_data)

    # Generate the final submission file
    print("Generating final submission...")
    stacker.predict_and_submit(meta_model, stacking_data)

    # -------------------------------------------------------------------------
    # 6. Final Validation
    # -------------------------------------------------------------------------
    print("\n=== Step 5: Final Submission Validation ===")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not created at {Config.SUBMISSION_PATH}"
        )

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission File Loaded. Shape: {sub_df.shape}")

    # Validate Dimensions
    # Rows should equal number of test images
    expected_rows = len(test_ids)
    if sub_df.shape[0] != expected_rows:
        raise AssertionError(
            f"Submission has {sub_df.shape[0]} rows, expected {expected_rows}"
        )

    # Columns should be ID + 120 breeds
    expected_cols = 1 + Config.NUM_CLASSES
    if sub_df.shape[1] != expected_cols:
        raise AssertionError(
            f"Submission has {sub_df.shape[1]} columns, expected {expected_cols}"
        )

    # Validate ID column
    if "id" not in sub_df.columns:
        raise AssertionError("Submission missing 'id' column")

    # Validate Probabilities
    # Select all columns except 'id'
    prob_cols = sub_df.columns.drop("id")
    probs = sub_df[prob_cols].values

    # Check if values are within [0, 1]
    if not (
        np.all(probs >= 0) and np.all(probs <= 1.0001)
    ):  # slightly loose for float precision
        raise AssertionError("Found probability values outside [0, 1]")

    # Check if probabilities sum to 1 (row-wise)
    row_sums = probs.sum(axis=1)
    # Allow for small floating point errors
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        raise AssertionError("Probabilities do not sum to 1.0")

    print("Submission file passed all validation checks.")
    print("\n=== Demo Execution Completed Successfully ===")


if __name__ == "__main__":
    main()
