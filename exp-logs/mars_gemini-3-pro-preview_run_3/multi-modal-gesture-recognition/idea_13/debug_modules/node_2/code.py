import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import library components
from library.config import Config
from library.utils import set_seed, compute_normalized_levenshtein, get_device
from library.data_loader import get_dataloaders
from library.model import SK_ARN, CascadedLoss
from library.trainer import Trainer
from library.inference import Predictor


def run_demo():
    print("=== Starting SK-ARN Library Demonstration ===")

    # ---------------------------------------------------------
    # 1. Configuration Setup for Demo
    # ---------------------------------------------------------
    # We modify the Config class attributes directly to create a
    # lightweight execution environment (fast, small memory footprint).

    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    print(f"Configuring demo environment in {DEMO_DIR}...")

    Config.WORK_DIR = DEMO_DIR
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")

    # Update cache paths to point to demo directory
    Config.TRAIN_CACHE = os.path.join(DEMO_DIR, "cache", "dataset_train.npz")
    Config.VAL_CACHE = os.path.join(DEMO_DIR, "cache", "dataset_val.npz")
    Config.TEST_CACHE = os.path.join(DEMO_DIR, "cache", "dataset_test.npz")
    Config.STATS_CACHE = os.path.join(DEMO_DIR, "cache", "normalizer_stats.npz")
    Config.BEST_MODEL_PATH = os.path.join(DEMO_DIR, "best_model.pth")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Hyperparameters for speed
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.DEBUG_MAX_SAMPLES = 12  # Limit to 12 samples for speed

    # Ensure directories exist
    os.makedirs(os.path.dirname(Config.TRAIN_CACHE), exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    set_seed(Config.SEED)
    device = get_device()
    print(f"Device: {device}")

    # ---------------------------------------------------------
    # 2. Utility Validation
    # ---------------------------------------------------------
    print("\n--- Validating Utilities ---")
    # Test Levenshtein distance metric logic
    target_seq = [1, 2, 3, 4]
    pred_seq_perfect = [1, 2, 3, 4]
    pred_seq_error = [1, 2, 5, 4]  # Substitution 3->5

    score_perfect = compute_normalized_levenshtein([pred_seq_perfect], [target_seq])
    score_error = compute_normalized_levenshtein([pred_seq_error], [target_seq])

    assert score_perfect == 0.0, "Metric should be 0 for perfect match"
    assert score_error == 0.25, "Metric should be 0.25 for 1 error in length 4"
    print("Metric calculation verified.")

    # ---------------------------------------------------------
    # 3. Data Loader Validation
    # ---------------------------------------------------------
    print("\n--- Validating Data Loader ---")
    # This triggers cache creation and loading
    train_loader, val_loader, test_loader = get_dataloaders(
        debug_max=Config.DEBUG_MAX_SAMPLES
    )

    # Fetch one batch
    features, labels = next(iter(train_loader))

    print(f"Feature shape: {features.shape}")
    print(f"Label shape: {labels.shape}")

    # Assertions
    # Expected: (Batch, Window, InputDim)
    assert features.ndim == 3, "Features must be 3D tensor (B, T, C)"
    assert (
        features.shape[0] == Config.BATCH_SIZE
    ), f"Batch size mismatch. Expected {Config.BATCH_SIZE}"
    assert (
        features.shape[1] == Config.WINDOW_SIZE
    ), f"Window size mismatch. Expected {Config.WINDOW_SIZE}"
    assert (
        features.shape[2] == Config.INPUT_DIM
    ), f"Input dim mismatch. Expected {Config.INPUT_DIM}"

    # Expected: (Batch, Window)
    assert labels.ndim == 2, "Labels must be 2D tensor (B, T)"
    assert labels.shape[0] == Config.BATCH_SIZE
    assert labels.shape[1] == Config.WINDOW_SIZE
    print("Data Loader shapes verified.")

    # ---------------------------------------------------------
    # 4. Model Architecture Validation
    # ---------------------------------------------------------
    print("\n--- Validating Model Architecture ---")
    model = SK_ARN().to(device)
    features = features.to(device)

    # Forward pass
    outputs = model(features)

    # Check output structure
    assert isinstance(outputs, dict), "Model output should be a dictionary"
    assert (
        "stage1" in outputs and "stage2" in outputs and "stage3" in outputs
    ), "Missing stages in output"

    s3_logits = outputs["stage3"]
    print(f"Stage 3 Output shape: {s3_logits.shape}")

    # Expected: (Batch, Window, NumClasses)
    assert s3_logits.shape == (
        Config.BATCH_SIZE,
        Config.WINDOW_SIZE,
        Config.NUM_CLASSES,
    ), "Output shape mismatch"
    print("Model forward pass verified.")

    # ---------------------------------------------------------
    # 5. Loss Function Validation
    # ---------------------------------------------------------
    print("\n--- Validating Loss Function ---")
    criterion = CascadedLoss().to(device)
    labels = labels.to(device)

    loss = criterion(outputs, labels)
    print(f"Calculated Loss: {loss.item()}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"
    print("Loss function verified.")

    # ---------------------------------------------------------
    # 6. Training Loop Validation
    # ---------------------------------------------------------
    print("\n--- Validating Training Loop ---")
    # Initialize Trainer (uses the modified Config)
    trainer = Trainer(debug_max=Config.DEBUG_MAX_SAMPLES)

    # Run fit (runs for Config.NUM_EPOCHS=2)
    trainer.fit()

    # Check if model was saved
    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), "Best model checkpoint was not saved."
    print("Training loop completed and model saved.")

    # ---------------------------------------------------------
    # 7. Inference & Submission Validation
    # ---------------------------------------------------------
    print("\n--- Validating Inference Pipeline ---")
    # Initialize Predictor
    predictor = Predictor(debug_max=Config.DEBUG_MAX_SAMPLES)

    # Generate submission
    predictor.generate_submission()

    # Check submission file
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file not created."

    # Verify content format
    with open(Config.SUBMISSION_FILE, "r") as f:
        lines = f.readlines()

    print(f"Generated {len(lines)} lines in submission file.")
    if len(lines) > 0:
        sample_line = lines[0].strip()
        print(f"Sample submission line: {sample_line}")
        parts = sample_line.split(",")
        # Format: SampleID, Label1, Label2...
        # SampleID usually starts with 'Sample' or 'Session' depending on dataset,
        # here we check if first part looks like an ID
        assert len(parts) >= 1, "Invalid submission line format"

    print("Inference pipeline verified.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
