import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import joblib

# Import from the provided library
from library.config import Config
from library.dataset import get_loaders
from library.training import run_fold, get_model
from library.stacking import GeometricStacking
from library.utils import seed_everything


def run_demo():
    print("==== Starting Cactus Classification Demo ====")

    # ---------------------------------------------------------
    # 1. Configuration Override for Fast Execution
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for demo...")

    # Set a specific working directory for this demo to avoid conflicts
    demo_work_dir = "./working/demo_execution"
    if os.path.exists(demo_work_dir):
        shutil.rmtree(demo_work_dir)
    os.makedirs(demo_work_dir, exist_ok=True)

    # Override Config attributes
    Config.WORK_DIR = demo_work_dir
    Config.SUBMISSION_DIR = os.path.join(demo_work_dir, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Update cache paths to point to the new working directory
    Config.CACHE_TRAIN_IMGS = os.path.join(Config.WORK_DIR, "cache_train_imgs.npy")
    Config.CACHE_TRAIN_LABELS = os.path.join(Config.WORK_DIR, "cache_train_labels.npy")
    Config.CACHE_TEST_IMGS = os.path.join(Config.WORK_DIR, "cache_test_imgs.npy")
    Config.CACHE_TEST_IDS = os.path.join(Config.WORK_DIR, "cache_test_ids.npy")
    Config.META_LEARNER_PATH = os.path.join(
        Config.WORK_DIR, "meta_learner_logreg.joblib"
    )

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Runtime constraints for speed
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 200  # Small subset for quick processing
    Config.EPOCHS = 2  # Minimal epochs
    Config.SWA_START_EPOCH = 1  # Start SWA immediately to test logic
    Config.N_FOLDS = 2  # Only 2 folds for demonstration
    Config.BATCH_SIZE = 16  # Smaller batch size
    Config.LOAD_CACHED_DATA = False  # Force processing from metadata CSVs

    # Limit to a single model architecture to save time
    # We use CactusResNet as it is standard and robust
    Config.MODEL_ARCHITECTURES = ["CactusResNet"]

    # Set random seed
    seed_everything(Config.SEED)

    print(f"  Working Directory: {Config.WORK_DIR}")
    print(f"  Model Architectures: {Config.MODEL_ARCHITECTURES}")
    print(
        f"  Epochs: {Config.EPOCHS}, Folds: {Config.N_FOLDS}, Debug Mode: {Config.DEBUG}"
    )

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("\n[2] Loading Data...")
    train_loader, val_loader, test_loader, test_ids = get_loaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=2,  # Reduced workers for small debug set
        debug=Config.DEBUG,
    )

    # Verification
    batch_img, batch_label = next(iter(train_loader))
    print(f"  Train Batch Shape: {batch_img.shape}")
    assert batch_img.shape == (
        Config.BATCH_SIZE,
        3,
        32,
        32,
    ), "Incorrect image batch shape"
    assert batch_label.shape == (Config.BATCH_SIZE,), "Incorrect label batch shape"
    assert (
        len(test_ids) == Config.DEBUG_SUBSET_SIZE
    ), "Incorrect number of test IDs loaded"
    print("  Data loading verification passed.")

    # ---------------------------------------------------------
    # 3. Model Training (Cross-Validation)
    # ---------------------------------------------------------
    print("\n[3] Starting Training Loop...")

    # We iterate through the defined architectures and folds
    for model_name in Config.MODEL_ARCHITECTURES:
        for fold in range(Config.N_FOLDS):
            print(f"\n  >> Training {model_name} - Fold {fold}")

            # Run the training for this fold
            results = run_fold(fold, model_name, train_loader, val_loader)

            # Verify outputs
            assert "best_auc" in results
            assert "swa_auc" in results

            # Verify checkpoints exist
            best_ckpt = Config.get_checkpoint_path(f"{model_name}_best", fold)
            swa_ckpt = Config.get_checkpoint_path(f"{model_name}_swa", fold)

            assert os.path.exists(best_ckpt), f"Best checkpoint not found: {best_ckpt}"
            assert os.path.exists(swa_ckpt), f"SWA checkpoint not found: {swa_ckpt}"
            print(f"  Fold {fold} completed. Best AUC: {results['best_auc']:.4f}")

    # ---------------------------------------------------------
    # 4. Stacking & Meta-Learning
    # ---------------------------------------------------------
    print("\n[4] Running Geometric Stacking...")

    stacker = GeometricStacking()

    # Generate Meta-Features (Predictions from base models)
    # This uses Test-Time Augmentation (TTA) internally
    print("  Generating meta-features (Validation and Test)...")
    X_val, y_val, X_test = stacker.generate_geometric_features(
        val_loader, test_loader, load_cached=False
    )

    # Verify feature shapes
    # Shape should be: (N_samples, N_models * N_folds * 2_stats)
    # 2 stats are Mean and Std from TTA
    expected_feature_dim = len(Config.MODEL_ARCHITECTURES) * Config.N_FOLDS * 2
    print(f"  Meta-feature shape (Val): {X_val.shape}")

    assert (
        X_val.shape[1] == expected_feature_dim
    ), f"Expected {expected_feature_dim} features, got {X_val.shape[1]}"
    assert (
        X_val.shape[0] == Config.DEBUG_SUBSET_SIZE
    ), f"Expected {Config.DEBUG_SUBSET_SIZE} validation samples, got {X_val.shape[0]}"

    # Train Meta-Learner
    print("  Training Meta-Learner...")
    clf = stacker.train_meta_learner(X_val, y_val)

    assert os.path.exists(
        Config.META_LEARNER_PATH
    ), "Meta-learner model file not saved."

    # ---------------------------------------------------------
    # 5. Final Inference & Submission
    # ---------------------------------------------------------
    print("\n[5] Generating Final Submission...")

    stacker.predict_stacking(X_test, test_ids)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"  Submission loaded. Shape: {df_sub.shape}")
    print(f"  Head:\n{df_sub.head()}")

    assert list(df_sub.columns) == [
        "id",
        "has_cactus",
    ], "Submission columns are incorrect."
    assert len(df_sub) == Config.DEBUG_SUBSET_SIZE, "Submission row count mismatch."
    assert (
        df_sub["has_cactus"].min() >= 0 and df_sub["has_cactus"].max() <= 1
    ), "Probabilities out of range."

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
