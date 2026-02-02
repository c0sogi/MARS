import os
import torch
import numpy as np
import shutil
import pandas as pd

# Import library modules
from library.config import Paths, DataConfig, TrainConfig, ModelConfig
from library.data_loader import get_dataloaders
from library.model import CKARFNet
from library.utils import set_seed, compute_levenshtein, rle_encode
import library.train as train_module
import library.predict as predict_module


def main():
    print("=== Starting Demonstration Script ===")

    # ---------------------------------------------------------
    # 1. Setup & Configuration Override
    # ---------------------------------------------------------
    # We override the configuration to run a fast demo in a separate directory
    demo_dir = "./working/demo_execution"

    # Clean up previous demo run if exists
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override Paths
    Paths.WORKING_DIR = demo_dir
    Paths.CACHE_DIR = os.path.join(demo_dir, "cache")
    Paths.CHECKPOINT_DIR = os.path.join(demo_dir, "checkpoints")
    Paths.SUBMISSION_DIR = os.path.join(demo_dir, "submission")
    Paths.SUBMISSION_FILE = os.path.join(Paths.SUBMISSION_DIR, "submission.csv")

    # Create directories
    os.makedirs(Paths.CACHE_DIR, exist_ok=True)
    os.makedirs(Paths.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Paths.SUBMISSION_DIR, exist_ok=True)

    # Override Data Config for speed
    DataConfig.DEBUG_SAMPLE_SIZE = 20  # Use only 20 samples

    # Override Train Config for speed
    TrainConfig.EPOCHS = 1
    TrainConfig.BATCH_SIZE = 4
    TrainConfig.PATIENCE = 1

    # Set seed for reproducibility
    set_seed(42)

    print(f"Configuration set. Working directory: {Paths.WORKING_DIR}")

    # ---------------------------------------------------------
    # 2. Data Loading Verification
    # ---------------------------------------------------------
    print("\n--- Verifying Data Loading ---")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=TrainConfig.BATCH_SIZE,
        num_workers=0,  # Use 0 workers for simple debugging
        debug_size=DataConfig.DEBUG_SAMPLE_SIZE,
    )

    # Fetch one batch
    batch = next(iter(train_loader))
    features = batch["features"]
    labels = batch["labels"]

    # Check dimensions
    # Features: (Batch, Time, InputDim)
    # InputDim = (20 joints * 3 coords * 3 derivatives) + 13 MFCC = 180 + 13 = 193
    expected_input_dim = 193
    assert features.dim() == 3, f"Expected 3D features, got {features.shape}"
    assert (
        features.shape[2] == expected_input_dim
    ), f"Expected input dim {expected_input_dim}, got {features.shape[2]}"
    assert labels.dim() == 2, f"Expected 2D labels (Batch, Time), got {labels.shape}"
    assert (
        features.shape[0] == TrainConfig.BATCH_SIZE
    ), f"Expected batch size {TrainConfig.BATCH_SIZE}, got {features.shape[0]}"

    print(f"Data Loader check passed. Feature shape: {features.shape}")

    # ---------------------------------------------------------
    # 3. Model Verification
    # ---------------------------------------------------------
    print("\n--- Verifying Model Architecture ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CKARFNet().to(device)

    # Move batch to device
    features_dev = features.to(device)

    # Forward pass
    outputs = model(features_dev)

    # Check output structure (Deep Supervision: Tuple of 3)
    assert isinstance(
        outputs, tuple
    ), "Model output should be a tuple (Deep Supervision)"
    assert len(outputs) == 3, f"Expected 3 stages of output, got {len(outputs)}"

    # Check shape of final stage: (Batch, Time, NumClasses)
    final_logits = outputs[-1]
    expected_classes = DataConfig.NUM_CLASSES  # 21
    assert final_logits.shape[0] == TrainConfig.BATCH_SIZE
    assert final_logits.shape[1] == features.shape[1]  # Time dimension preserved
    assert final_logits.shape[2] == expected_classes

    print(f"Model check passed. Output shape: {final_logits.shape}")

    # ---------------------------------------------------------
    # 4. Metric Verification
    # ---------------------------------------------------------
    print("\n--- Verifying Metrics ---")
    # Test Levenshtein
    # Sequence A: [1, 2, 3]
    # Sequence B: [1, 2, 3] -> Dist 0
    # Sequence C: [1, 2] -> Dist 1 (Deletion)

    score_perfect = compute_levenshtein([[1, 2, 3]], [[1, 2, 3]])
    assert score_perfect == 0.0, f"Expected 0.0 for perfect match, got {score_perfect}"

    score_diff = compute_levenshtein([[1, 2]], [[1, 2, 3]])
    # Dist is 1, Length is 3 -> 1/3
    expected_score = 1.0 / 3.0
    assert (
        abs(score_diff - expected_score) < 1e-5
    ), f"Expected {expected_score}, got {score_diff}"

    print("Metric check passed.")

    # ---------------------------------------------------------
    # 5. Training Loop Execution
    # ---------------------------------------------------------
    print("\n--- Running Training Loop (1 Epoch) ---")
    # We call the run_training function from the library.
    # It uses the global configuration we overrode earlier.

    try:
        train_module.run_training()
        print("Training execution completed successfully.")
    except Exception as e:
        raise RuntimeError(f"Training failed: {e}")

    # Verify checkpoint creation
    expected_ckpt = os.path.join(Paths.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(expected_ckpt), "Checkpoint file was not created."
    print(f"Checkpoint verified at {expected_ckpt}")

    # ---------------------------------------------------------
    # 6. Inference Execution
    # ---------------------------------------------------------
    print("\n--- Running Inference ---")

    try:
        predict_module.generate_submission(
            device_str="cuda" if torch.cuda.is_available() else "cpu", num_workers=0
        )
        print("Inference execution completed successfully.")
    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}")

    # Verify submission file
    assert os.path.exists(Paths.SUBMISSION_FILE), "Submission file was not created."

    # Check content
    with open(Paths.SUBMISSION_FILE, "r") as f:
        lines = f.readlines()

    print(f"Submission file created with {len(lines)} lines.")
    if len(lines) > 0:
        print(f"Sample line: {lines[0].strip()}")

        # Verify format: SessionID,Label1,Label2...
        parts = lines[0].strip().split(",")
        assert len(parts) >= 1, "Line format incorrect (empty?)"
        assert parts[0].startswith("Session") or parts[0].startswith(
            "Sample"
        ), f"Unexpected ID format: {parts[0]}"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
