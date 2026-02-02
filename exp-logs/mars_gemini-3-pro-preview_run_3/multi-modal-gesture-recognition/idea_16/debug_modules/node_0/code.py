import os
import sys
import shutil
import numpy as np
import torch
import pandas as pd

# Import from the provided library
from library.config import Config
from library.utils import (
    set_seed,
    rle_encode_predictions,
    compute_levenshtein_distance,
    evaluate_predictions,
)
from library.data_loader import GestureDataset, get_dataloaders
from library.model import RDKRN
from library.loss import CascadedLoss
from library.train import run_training
from library.inference import run_inference


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup Configuration for Demo
    # We use a specific directory for this execution to ensure a clean state
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)

    # Monkey-patch Config to use the demo directory
    Config.WORK_DIR = demo_dir
    Config.CACHE_DIR = os.path.join(demo_dir, "cache")
    Config.MODEL_SAVE_PATH = os.path.join(demo_dir, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Create directories
    Config.setup_directories()

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print(f"Working directory set to: {Config.WORK_DIR}")

    # 2. Verify Utility Functions
    print("\n--- Verifying Utilities ---")

    # Test RLE Encoding
    # Sequence: 1, 1, 1, 0, 0, 2, 2, 0, 3 -> [1, 2, 3] (0 is background)
    raw_preds = [1, 1, 1, 0, 0, 2, 2, 0, 3]
    encoded = rle_encode_predictions(raw_preds, background_id=0)
    expected = [1, 2, 3]
    assert encoded == expected, f"RLE Failed. Expected {expected}, got {encoded}"
    print("RLE Encoding: OK")

    # Test Levenshtein Distance
    seq_a = [1, 2, 3]
    seq_b = [1, 3]  # Deletion of 2
    dist = compute_levenshtein_distance(seq_a, seq_b)
    assert dist == 1.0, f"Levenshtein Distance Failed. Expected 1.0, got {dist}"
    print("Levenshtein Distance: OK")

    # 3. Verify Data Loading
    print("\n--- Verifying Data Loader ---")
    # Use a very small subset (limit_samples=5) for speed
    # This will trigger processing and caching
    print("Initializing Train Dataset (Subset)...")
    train_ds = GestureDataset(Config.TRAIN_CSV, split="train", limit_samples=5)

    if len(train_ds) == 0:
        print(
            "Warning: Dataset is empty (possibly due to very small subset or filtering)."
        )
    else:
        # Fetch one sample
        sample = train_ds[0]
        features = sample["features"]
        targets = sample["targets"]

        print(
            f"Sample Features Shape: {features.shape}"
        )  # Expected: (WindowSize, InputDim)
        print(f"Sample Targets Shape: {targets.shape}")  # Expected: (WindowSize,)

        # Validation
        assert features.shape == (
            Config.WINDOW_SIZE,
            Config.INPUT_DIM,
        ), f"Feature shape mismatch. Expected {(Config.WINDOW_SIZE, Config.INPUT_DIM)}, got {features.shape}"
        assert targets.shape == (
            Config.WINDOW_SIZE,
        ), f"Target shape mismatch. Expected {(Config.WINDOW_SIZE,)}, got {targets.shape}"
        assert isinstance(features, torch.Tensor), "Features should be a torch.Tensor"

    print("Data Loader: OK")

    # 4. Verify Model Architecture
    print("\n--- Verifying Model ---")
    model = RDKRN()
    model.eval()

    # Create dummy batch: (Batch=2, Time=64, InputDim=193)
    dummy_input = torch.randn(2, Config.WINDOW_SIZE, Config.INPUT_DIM)

    # Forward pass
    with torch.no_grad():
        outputs = model(dummy_input)

    # Model returns a list of logits from 3 stages
    assert isinstance(outputs, list), "Model output should be a list"
    assert (
        len(outputs) == 3
    ), f"Model should return outputs for 3 stages, got {len(outputs)}"

    for i, out in enumerate(outputs):
        # Shape: (Batch, Time, NumClasses)
        expected_shape = (2, Config.WINDOW_SIZE, Config.NUM_CLASSES)
        assert (
            out.shape == expected_shape
        ), f"Stage {i+1} output shape mismatch. Expected {expected_shape}, got {out.shape}"

    print("Model Architecture: OK")

    # 5. Verify Loss Function
    print("\n--- Verifying Loss Function ---")
    criterion = CascadedLoss()

    # Dummy targets: (Batch=2, Time=64)
    dummy_targets = torch.randint(0, Config.NUM_CLASSES, (2, Config.WINDOW_SIZE))

    # Compute loss
    loss, loss_dict = criterion(outputs, dummy_targets)

    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.dim() == 0, "Loss should be scalar"
    assert "total_loss" in loss_dict, "Loss dict missing total_loss"
    assert "loss_ce_stage1" in loss_dict, "Loss dict missing stage 1 CE"
    assert "loss_mse_stage2" in loss_dict, "Loss dict missing stage 2 MSE"

    print(f"Calculated Loss: {loss.item():.4f}")
    print("Loss Function: OK")

    # 6. Verify Training Loop
    print("\n--- Running Training Demo ---")
    # Run for 1 epoch with a small subset
    # limit_samples=10 ensures it runs very fast
    run_training(limit_samples=10, num_epochs=1)

    # Check if model was saved
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Model checkpoint successfully saved to {Config.MODEL_SAVE_PATH}")
    else:
        raise FileNotFoundError("Training completed but model checkpoint not found.")

    print("Training Loop: OK")

    # 7. Verify Inference Pipeline
    print("\n--- Running Inference Demo ---")
    # Run inference on a small subset of test data
    run_inference(limit_samples=5)

    # Check submission file
    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"Submission file successfully saved to {Config.SUBMISSION_PATH}")

        # Validate content format
        df_sub = pd.read_csv(Config.SUBMISSION_PATH, header=None)
        print("Submission Head:")
        print(df_sub.head())

        # Check basic structure (SessionID, predictions)
        # Note: Some predictions might be NaN/empty if the model predicts only background
        assert (
            df_sub.shape[1] >= 1
        ), "Submission should have at least one column (SessionID)"
    else:
        raise FileNotFoundError("Inference completed but submission file not found.")

    print("Inference Pipeline: OK")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
