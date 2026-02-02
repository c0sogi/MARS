import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_device, calculate_roc_auc
from library.dataset import get_dataloaders
from library.train import Trainer
from library.stacking import fit_stacking_model, predict_stacking_model


def run_demo():
    print("=== Starting Whale Detection Pipeline Demo ===")

    # ---------------------------------------------------------
    # 1. Setup and Configuration Override
    # ---------------------------------------------------------
    print("\n[Step 1] Configuring environment for demo execution...")

    # Set reproducible seed
    seed_everything(Config.SEED)

    # Override Config for speed and demo purposes
    # We use a separate working directory to avoid messing with existing experiments
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Reduce compute load for the demo
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Use smaller/fewer models for the demonstration
    # We use two models to demonstrate the stacking logic
    Config.MODEL_NAMES = ["resnet18", "tf_efficientnet_b0"]

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Models to train: {Config.MODEL_NAMES}")
    print(f"Epochs: {Config.EPOCHS}, Batch Size: {Config.BATCH_SIZE}")

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("\n[Step 2] Loading Data (Debug Mode)...")

    # get_dataloaders(debug=True) loads a small subset (head of the dataframes)
    # load_cached_data=False ensures we process the raw audio for this demo
    # (or verify the processing logic) rather than loading potentially stale cache.
    train_loader, val_loader, test_loader, test_clips = get_dataloaders(
        debug=True, load_cached_data=False
    )

    # Verification
    assert len(train_loader) > 0, "Train loader is empty."
    assert len(val_loader) > 0, "Val loader is empty."
    assert len(test_loader) > 0, "Test loader is empty."

    # Check data shapes from a single batch
    sample_batch, sample_targets = next(iter(train_loader))
    print(f"Sample batch shape: {sample_batch.shape}")  # Should be (B, 1, 128, T)
    print(f"Sample target shape: {sample_targets.shape}")

    assert sample_batch.dim() == 4, "Input must be 4D (B, C, F, T)"
    assert sample_batch.size(1) == 1, "Input must have 1 channel"

    # ---------------------------------------------------------
    # 3. Ensemble Training
    # ---------------------------------------------------------
    print("\n[Step 3] Training Ensemble...")

    device = get_device()
    val_meta_features = []
    test_meta_features = []

    # Collect validation targets for the meta-learner
    val_targets = []
    for _, target in val_loader:
        val_targets.extend(target.numpy())
    val_targets = np.array(val_targets)

    for model_name in Config.MODEL_NAMES:
        print(f"\n--- Processing Model: {model_name} ---")

        # Initialize Trainer
        # We pass debug=False here because we manually set Config.EPOCHS=1.
        # If we passed debug=True to Trainer, it might override epochs to 2 (based on library code).
        trainer = Trainer(model_name, train_loader, val_loader, device, debug=False)

        # Train
        best_auc = trainer.fit()

        # Verify training produced a model file
        assert os.path.exists(
            trainer.best_model_path
        ), f"Model checkpoint not found at {trainer.best_model_path}"

        # Generate Predictions
        print(f"Generating predictions for {model_name}...")
        val_preds = trainer.predict(val_loader)
        test_preds = trainer.predict(test_loader)

        # Verify prediction shapes
        assert len(val_preds) == len(
            val_targets
        ), f"Val preds length {len(val_preds)} mismatch with targets {len(val_targets)}"
        assert len(test_preds) == len(
            test_clips
        ), f"Test preds length {len(test_preds)} mismatch with clips {len(test_clips)}"

        val_meta_features.append(val_preds)
        test_meta_features.append(test_preds)

    # ---------------------------------------------------------
    # 4. Stacking (Meta-Learning)
    # ---------------------------------------------------------
    print("\n[Step 4] Stacking Models...")

    # Stack features: (N_samples, N_models)
    X_val = np.column_stack(val_meta_features)
    X_test = np.column_stack(test_meta_features)
    y_val = val_targets

    print(f"Meta-feature shape (Val): {X_val.shape}")
    print(f"Meta-feature shape (Test): {X_test.shape}")

    # Train Meta-Learner
    meta_learner = fit_stacking_model(X_val, y_val, save_dir=Config.WORKING_DIR)

    # Validate Meta-Learner on Val set (Sanity Check)
    val_final_preds = predict_stacking_model(meta_learner, X_val)
    meta_auc = calculate_roc_auc(y_val, val_final_preds)
    print(f"Meta-Learner Validation AUC: {meta_auc:.4f}")

    # Generate Final Test Predictions
    final_test_preds = predict_stacking_model(meta_learner, X_test)

    # Verify probabilities are in [0, 1]
    assert np.all(final_test_preds >= 0) and np.all(
        final_test_preds <= 1
    ), "Predictions must be probabilities between 0 and 1"

    # ---------------------------------------------------------
    # 5. Submission Generation
    # ---------------------------------------------------------
    print("\n[Step 5] Generating Submission...")

    submission = pd.DataFrame({"clip": test_clips, "probability": final_test_preds})

    # Verify submission format
    assert submission.shape[1] == 2, "Submission must have 2 columns"
    assert (
        "clip" in submission.columns and "probability" in submission.columns
    ), "Submission columns must be 'clip' and 'probability'"

    # Save
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    # Final check
    if os.path.exists(Config.SUBMISSION_PATH):
        print("=== Demo Completed Successfully ===")
    else:
        raise FileNotFoundError("Submission file was not created.")


if __name__ == "__main__":
    run_demo()
