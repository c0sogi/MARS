import os
import sys
import torch
import pandas as pd
import numpy as np

# Ensure we can import from the library directory
sys.path.append(os.getcwd())

from library.utils import Config, set_seed
from library.dataset import get_dataloaders
from library.model import CactusResNet
from library.engine import run


def main():
    # 1. Setup and Configuration
    # We enable debug mode to limit the dataset size (500 samples) and epochs (2) for speed.
    config = Config(seed=42, debug=True, batch_size=32)
    set_seed(config.SEED)

    print("=== Configuration ===")
    print(config)
    print(f"Device: {config.DEVICE}")
    print("-" * 20)

    # 2. Demonstrate Data Loading and Verification
    print("\n=== Data Loading Verification ===")
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # Fetch a single batch to verify shapes and types
    try:
        images, labels = next(iter(train_loader))
        print(f"Batch Image Shape: {images.shape}")
        print(f"Batch Label Shape: {labels.shape}")

        # Assertions
        assert images.dim() == 4, "Images should be 4D tensors (B, C, H, W)"
        assert images.size(1) == 3, "Images should have 3 channels (RGB)"
        assert images.size(2) == 32 and images.size(3) == 32, "Images should be 32x32"
        assert labels.dim() == 1, "Labels should be 1D tensors"
        assert (
            images.size(0) == config.BATCH_SIZE
        ), f"Batch size mismatch. Expected {config.BATCH_SIZE}"

        print("Data Loading checks passed.")
    except StopIteration:
        print("Error: DataLoader is empty.")
        sys.exit(1)
    except AssertionError as e:
        print(f"Data Assertion Failed: {e}")
        sys.exit(1)

    # 3. Demonstrate Model Instantiation and Forward Pass
    print("\n=== Model Verification ===")
    model = CactusResNet(num_classes=config.NUM_CLASSES)
    model.to(config.DEVICE)

    # Move batch to device
    images = images.to(config.DEVICE)

    # Forward pass
    try:
        outputs = model(images)
        print(f"Model Output Shape: {outputs.shape}")

        # Assertions
        assert outputs.dim() == 2, "Model output should be 2D (Batch, Num_Classes)"
        assert outputs.size(0) == config.BATCH_SIZE, "Output batch size mismatch"
        assert outputs.size(1) == config.NUM_CLASSES, "Output class count mismatch"

        print("Model Forward Pass checks passed.")
    except Exception as e:
        print(f"Model execution failed: {e}")
        sys.exit(1)

    # 4. Run Full Training and Inference Pipeline
    print("\n=== Running Full Training Pipeline ===")
    # The run function encapsulates training loop, validation, and prediction
    try:
        run(config)
        print("Pipeline execution completed.")
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        sys.exit(1)

    # 5. Verify Submission Artifacts
    print("\n=== Submission Verification ===")
    if not os.path.exists(config.SUBMISSION_PATH):
        print(f"Error: Submission file not found at {config.SUBMISSION_PATH}")
        sys.exit(1)

    try:
        df_sub = pd.read_csv(config.SUBMISSION_PATH)
        print(f"Submission shape: {df_sub.shape}")
        print(f"Columns: {list(df_sub.columns)}")

        # Assertions
        expected_cols = ["id", "has_cactus"]
        assert (
            list(df_sub.columns) == expected_cols
        ), f"Columns mismatch. Expected {expected_cols}"

        # Check if probabilities are within [0, 1]
        probs = df_sub["has_cactus"]
        assert (
            probs.min() >= 0.0 and probs.max() <= 1.0
        ), "Probabilities out of range [0, 1]"

        # Check ID count matches debug subset size if applied to test set
        # Note: Config.DEBUG_SUBSET_SIZE applies to test set in dataset.py
        if config.DEBUG:
            assert (
                len(df_sub) == config.DEBUG_SUBSET_SIZE
            ), f"Submission rows ({len(df_sub)}) do not match debug subset size ({config.DEBUG_SUBSET_SIZE})"

        print("Submission verification passed successfully.")

    except AssertionError as e:
        print(f"Submission Assertion Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error verifying submission: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
