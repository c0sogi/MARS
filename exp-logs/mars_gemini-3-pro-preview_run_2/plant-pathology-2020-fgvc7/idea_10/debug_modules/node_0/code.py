import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import library components
from library.config import CFG
from library.utils import seed_everything, calculate_metric, get_class_weights
from library.dataset import process_data, get_loaders, AppleDataset
from library.models import AppleNet
from library.trainer import run_fold, train_one_epoch, valid_one_epoch
from library.stacking import (
    train_meta_learner,
    inference_meta_learner,
    create_submission,
    rank_normalize,
)


def demo_pipeline():
    print("=== Starting Apple Disease Detection Pipeline Demo ===\n")

    # 1. Configuration Setup for Demo
    # We override specific CFG parameters to ensure the demo runs quickly and uses a separate directory.
    print("1. Configuring Environment...")
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override CFG settings
    CFG.working_dir = demo_dir
    CFG.output_dir = demo_dir
    CFG.submission_dir = demo_dir
    CFG.submission_path = os.path.join(demo_dir, "submission.csv")

    # Update cache paths to point to demo directory
    CFG.train_cache_path = os.path.join(demo_dir, "train_cache.parquet")
    CFG.val_cache_path = os.path.join(demo_dir, "val_cache.parquet")
    CFG.test_cache_path = os.path.join(demo_dir, "test_cache.parquet")

    # Reduce compute load for demo
    CFG.epochs = 1
    CFG.batch_size = 4
    CFG.num_workers = 0  # Avoid multiprocessing overhead for tiny demo
    CFG.debug = True

    # Set seed
    seed_everything(CFG.seed)
    print(f"   Working directory set to: {CFG.working_dir}")
    print("   Configuration updated for speed (Epochs=1, Batch=4).")

    # 2. Data Processing
    print("\n2. Processing Data...")
    # Load and process metadata
    # We use the existing metadata files in ./metadata
    train_df = process_data(
        CFG.train_metadata_path, CFG.train_cache_path, load_cached_data=False
    )
    val_df = process_data(
        CFG.val_metadata_path, CFG.val_cache_path, load_cached_data=False
    )
    test_df = process_data(
        CFG.test_metadata_path, CFG.test_cache_path, load_cached_data=False
    )

    # Create a tiny subset for demonstration purposes
    subset_size = 16
    train_subset = train_df.head(subset_size).copy()
    val_subset = val_df.head(subset_size).copy()
    test_subset = test_df.head(subset_size).copy()

    print(f"   Data loaded. Subset size for demo: {len(train_subset)} samples.")

    # Verify target processing (decomposition into binary tasks)
    assert "target_rust" in train_subset.columns, "Target 'rust' not created."
    assert "target_scab" in train_subset.columns, "Target 'scab' not created."
    print("   Target columns 'target_rust' and 'target_scab' verified.")

    # 3. Data Loaders
    print("\n3. Initializing Data Loaders...")
    # Use a smaller image size for speed
    img_size = 224
    train_loader, val_loader, test_loader = get_loaders(
        train_subset, val_subset, test_subset, img_size, batch_size=CFG.batch_size
    )

    # Verify Loader Output
    images, labels, ids = next(iter(train_loader))
    print(f"   Batch shapes - Images: {images.shape}, Labels: {labels.shape}")

    # Assertions
    assert images.shape == (
        CFG.batch_size,
        3,
        img_size,
        img_size,
    ), "Incorrect image batch shape"
    assert labels.shape == (
        CFG.batch_size,
        2,
    ), "Incorrect label batch shape (should be rust, scab)"
    assert len(ids) == CFG.batch_size, "Incorrect number of IDs"
    print("   DataLoaders functioning correctly.")

    # 4. Model Initialization
    print("\n4. Initializing Model...")
    # We use one of the backbones defined in config, but we'll use a smaller one if available or just the first one.
    # To save download time/memory, we'll stick to the first one but run minimal steps.
    model_name = CFG.backbones[
        1
    ]  # convnext_base is usually slightly lighter or similar, let's pick index 1
    print(f"   Instantiating {model_name}...")

    model = AppleNet(model_name=model_name, pretrained=True)
    model.to(CFG.device)

    # Verify Forward Pass
    with torch.no_grad():
        dummy_input = torch.randn(2, 3, img_size, img_size).to(CFG.device)
        output = model(dummy_input)

    print(f"   Model Output Shape: {output.shape}")
    assert output.shape == (2, 2), "Model output shape mismatch (Batch=2, Classes=2)"
    print("   Model instantiated and forward pass verified.")

    # 5. Training Loop (Simulation)
    print("\n5. Running Training Loop (1 Epoch, Fold 0)...")

    # We will use run_fold but with our subset dataframes
    # run_fold saves the model to CFG.output_dir
    best_score = run_fold(
        fold=0,
        train_df=train_subset,
        val_df=val_subset,
        model_name=model_name,
        img_size=img_size,
    )

    print(f"   Training finished. Best Score: {best_score}")

    # Verify model artifact creation
    safe_model_name = model_name.replace(".", "_")
    expected_model_path = os.path.join(
        CFG.output_dir, f"best_model_{safe_model_name}_fold_0.pth"
    )
    assert os.path.exists(
        expected_model_path
    ), f"Model file not found at {expected_model_path}"
    print(f"   Model artifact verified at: {expected_model_path}")

    # 6. Metric Calculation Utility
    print("\n6. Verifying Metric Calculation...")
    # Create dummy ground truth and predictions
    y_true = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    y_pred = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.9], [0.1, 0.2]])

    metric = calculate_metric(y_true, y_pred)
    print(f"   Calculated ROC AUC: {metric:.4f}")
    assert 0 <= metric <= 1, "Metric out of bounds"

    # 7. Stacking / Meta-Learner
    print("\n7. Stacking & Meta-Learner Pipeline...")

    # Simulate OOF predictions for 2 models
    # Shape: (N_samples, 2_classes)
    n_samples = len(train_subset)
    targets = train_subset[["target_rust", "target_scab"]].values

    # Generate synthetic OOF predictions
    oof_preds_model_1 = np.random.rand(n_samples, 2)
    oof_preds_model_2 = np.random.rand(n_samples, 2)

    oof_dict = {"model_1": oof_preds_model_1, "model_2": oof_preds_model_2}

    # Train Meta Learner
    print("   Training Meta-Learner...")
    meta_models = train_meta_learner(oof_dict, targets)
    assert (
        "rust" in meta_models and "scab" in meta_models
    ), "Meta-learner dictionary missing keys"

    # Simulate Test Predictions
    n_test = len(test_subset)
    test_preds_model_1 = np.random.rand(n_test, 2)
    test_preds_model_2 = np.random.rand(n_test, 2)

    test_preds_dict = {"model_1": test_preds_model_1, "model_2": test_preds_model_2}

    # Inference Meta Learner
    print("   Running Meta-Learner Inference...")
    rust_probs, scab_probs = inference_meta_learner(meta_models, test_preds_dict)

    assert len(rust_probs) == n_test, "Rust probabilities length mismatch"
    assert len(scab_probs) == n_test, "Scab probabilities length mismatch"

    # Create Submission
    print("   Creating Submission...")
    submission_df = create_submission(test_subset, rust_probs, scab_probs)

    # Verify Submission
    assert os.path.exists(CFG.submission_path), "Submission file not created"
    assert submission_df.shape == (
        n_test,
        5,
    ), f"Submission shape mismatch: {submission_df.shape}"
    expected_cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    assert list(submission_df.columns) == expected_cols, "Submission columns mismatch"

    # Check probability constraints (sum isn't necessarily 1 for multi-label logic derived from binary,
    # but individual probs should be 0-1. The reconstruction logic ensures healthy/multi/rust/scab
    # partition the space based on the two binary vars).
    # Logic check: healthy + multiple + rust + scab should sum to 1.0 for every row
    row_sums = submission_df[["healthy", "multiple_diseases", "rust", "scab"]].sum(
        axis=1
    )
    assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1.0"

    print("   Submission verified correctly.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    demo_pipeline()
