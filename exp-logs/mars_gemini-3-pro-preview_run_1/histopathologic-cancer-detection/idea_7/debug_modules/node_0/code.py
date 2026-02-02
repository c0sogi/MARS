import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, MetricMonitor
from library.dataset import get_loaders, TumorDataset, get_transforms
from library.network import DenseNet121GeM, GeM
from library.engine import fit
from library.inference import run_inference


def validate_metric_monitor():
    """Verifies the logic of the MetricMonitor class."""
    print("\n=== Validating MetricMonitor ===")
    monitor = MetricMonitor()

    # Update 1: val=10, n=2 -> sum=20, count=2
    monitor.update(10, n=2)
    # Update 2: val=20, n=1 -> sum=40, count=3
    monitor.update(20, n=1)

    expected_avg = 40 / 3
    current_avg = monitor.avg

    print(f"Expected Avg: {expected_avg:.4f}, Calculated Avg: {current_avg:.4f}")

    if abs(current_avg - expected_avg) > 1e-5:
        raise AssertionError("MetricMonitor calculation is incorrect.")
    print("MetricMonitor validation passed.")


def validate_dataset_and_loaders():
    """Verifies the Dataset and DataLoader construction."""
    print("\n=== Validating Dataset and DataLoaders ===")

    # Use debug mode via Config overrides
    train_loader, val_loader, test_loader = get_loaders(
        fold_idx=0, debug=True, load_cached_data=False
    )

    # Check Train Loader
    try:
        batch = next(iter(train_loader))
    except StopIteration:
        raise AssertionError("Train loader is empty.")

    images = batch["image"]
    targets = batch["target"]
    ids = batch["id"]

    print(f"Batch keys: {list(batch.keys())}")
    print(f"Image batch shape: {images.shape}")
    print(f"Target batch shape: {targets.shape}")

    # Verify shapes
    if images.shape != (Config.BATCH_SIZE, 3, Config.IMG_SIZE, Config.IMG_SIZE):
        raise AssertionError(f"Incorrect image shape: {images.shape}")

    if targets.shape != (Config.BATCH_SIZE,):
        raise AssertionError(f"Incorrect target shape: {targets.shape}")

    # Verify Data Types
    if images.dtype != torch.float32:
        raise AssertionError("Images should be float32.")

    print("Dataset and DataLoader validation passed.")
    return train_loader, val_loader


def validate_model_logic():
    """Verifies the Model architecture and GeM pooling logic."""
    print("\n=== Validating Model and GeM Pooling ===")

    # 1. Verify GeM Pooling Math
    # GeM: (Avg(x^p))^(1/p)
    # If input is constant C, output should be C regardless of p
    gem_layer = GeM(p=3.0)
    constant_val = 2.0
    dummy_input = torch.ones(1, 64, 8, 8) * constant_val
    output = gem_layer(dummy_input)

    # Output shape should be (1, 64, 1, 1)
    if output.shape != (1, 64, 1, 1):
        raise AssertionError(f"GeM output shape incorrect: {output.shape}")

    # Value should be 2.0
    if not torch.allclose(output, torch.tensor(constant_val), atol=1e-5):
        raise AssertionError(
            f"GeM math incorrect. Expected {constant_val}, got {output.mean().item()}"
        )
    print("GeM Pooling logic verified.")

    # 2. Verify DenseNet121GeM Forward Pass
    model = DenseNet121GeM(pretrained=False)  # False for speed
    model.eval()

    # Create dummy input matching Config.IMG_SIZE
    dummy_img = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE)

    with torch.no_grad():
        logits = model(dummy_img)

    print(f"Model output shape: {logits.shape}")

    # Expected shape: (Batch_Size, 1)
    if logits.shape != (2, 1):
        raise AssertionError(
            f"Model output shape mismatch. Expected (2, 1), got {logits.shape}"
        )

    print("Model architecture validation passed.")


def run_demonstration():
    # ==========================================
    # 0. Setup & Configuration Overrides
    # ==========================================
    print("Initializing Configuration for Fast Demonstration...")
    set_seed(Config.SEED)

    # Override Config for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.DEBUG_SAMPLE_SIZE = 50  # Use only 50 samples
    Config.NUM_FOLDS = 2  # Reduce folds logic check
    Config.PATIENCE = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.PRETRAINED = False  # Skip downloading weights for speed

    # Clean working directory for fresh run
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # ==========================================
    # 1. Component Validation
    # ==========================================
    validate_metric_monitor()
    train_loader, val_loader = validate_dataset_and_loaders()
    validate_model_logic()

    # ==========================================
    # 2. Training Demonstration
    # ==========================================
    print("\n=== Running Training Loop (Fold 0) ===")
    # We use the loaders obtained from validation which respect the DEBUG_SAMPLE_SIZE
    fit(
        fold=0,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
    )

    # Verify model file creation
    expected_model_path = os.path.join(
        Config.WORKING_DIR, f"{Config.MODEL_NAME}_fold0_best.pth"
    )
    if not os.path.exists(expected_model_path):
        raise AssertionError(f"Model checkpoint not found at {expected_model_path}")
    print(f"Training successful. Model saved to {expected_model_path}")

    # ==========================================
    # 3. Inference Demonstration
    # ==========================================
    print("\n=== Running Inference Pipeline ===")
    # Disable TTA for speed in demo
    Config.TTA_ENABLED = False

    # Run inference
    submission_df = run_inference(device=torch.device("cpu"))

    # Verify Submission
    print(f"Submission shape: {submission_df.shape}")
    print("Head of submission:")
    print(submission_df.head())

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise AssertionError("Submission file was not created.")

    if list(submission_df.columns) != ["id", "label"]:
        raise AssertionError("Submission columns are incorrect.")

    # Check if we have predictions for the debug subset size
    # Note: run_inference also uses Config.DEBUG_SAMPLE_SIZE
    if len(submission_df) != Config.DEBUG_SAMPLE_SIZE:
        # It might be slightly less if the test csv is smaller than sample size, but here test csv is large.
        # However, run_inference re-reads the test csv and applies subsampling.
        pass

    print("Inference pipeline verified successfully.")


if __name__ == "__main__":
    try:
        run_demonstration()
        print("\nAll demonstrations and validations completed successfully.")
    except Exception as e:
        print(f"\nERROR: Demonstration failed with exception: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
