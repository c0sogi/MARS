import os
import sys
import torch
import numpy as np
import pandas as pd

# ==========================================
# 1. PRE-IMPORT CONFIGURATION PATCHING
# ==========================================
# We patch the configuration before importing dependent modules to ensure
# the demonstration runs quickly (fewer epochs, smaller batches).
import library.config

print(f"Default Epochs: {library.config.NUM_EPOCHS}")
library.config.NUM_EPOCHS = 2  # Run only 2 epochs for demonstration
library.config.BATCH_SIZE = 8  # Small batch size for debug
library.config.NUM_FOLDS = 2  # Set valid number of folds
print(f"Patched Epochs: {library.config.NUM_EPOCHS}")

# ==========================================
# 2. IMPORTS
# ==========================================
from library.utils import set_seed, log_metric
from library.data import process_and_cache_data, get_data_loaders
from library.model import QCWBN
from library.trainer import run_fold

# ==========================================
# 3. DEMONSTRATION FUNCTIONS
# ==========================================


def demo_utils():
    """Demonstrates utility functions."""
    print("\n[Demo] Utils")
    set_seed(42)
    log_metric("Demo", "TestMetric", 0.12345, epoch=1)
    print("Utils verification passed.")


def demo_data_processing():
    """Demonstrates data processing and caching."""
    print("\n[Demo] Data Processing")

    # Force processing from scratch to verify logic
    # This reads from ./input/train.json and ./input/test.json
    data = process_and_cache_data(load_cached_data=False)

    # Verify keys
    expected_keys = [
        "train_images",
        "train_angles",
        "train_labels",
        "train_ids",
        "test_images",
        "test_angles",
        "test_ids",
        "stats",
    ]
    for key in expected_keys:
        assert key in data, f"Missing key in processed data: {key}"

    # Verify Shapes
    # train_images should be (N, 75, 75, 2)
    img_shape = data["train_images"].shape
    assert len(img_shape) == 4
    assert img_shape[1:] == (75, 75, 2)

    # Verify Stats
    stats = data["stats"]
    assert "min_b1" in stats
    assert "angle_mean" in stats

    print(f"Processed {img_shape[0]} training images.")
    print("Data processing verification passed.")


def demo_data_loaders():
    """Demonstrates DataLoader creation and batch retrieval."""
    print("\n[Demo] Data Loaders")

    # Use debug=True to get a small subset (100 samples)
    train_loader, val_loader, test_loader, test_ids = get_data_loaders(
        fold_index=0, debug=True
    )

    # Fetch one batch
    images, angles, labels = next(iter(train_loader))

    # Check dimensions
    # Images should be (Batch, 3, 75, 75) - 3 channels because of derived mean band
    assert images.dim() == 4
    assert images.size(1) == 3
    assert images.size(2) == 75
    assert images.size(3) == 75

    # Angles and Labels
    assert angles.dim() == 1
    assert labels.dim() == 1

    print(f"Batch loaded. Images: {images.shape}, Labels: {labels.shape}")
    print("Data loader verification passed.")


def demo_model_architecture():
    """Demonstrates model instantiation and forward pass."""
    print("\n[Demo] Model Architecture")

    model = QCWBN()
    model.eval()

    # Create dummy input matching the DataLoader output
    # (Batch, 3, 75, 75)
    dummy_images = torch.randn(4, 3, 75, 75)
    dummy_angles = torch.randn(4)

    with torch.no_grad():
        logits = model(dummy_images, dummy_angles)

    # Output should be (Batch, 1)
    assert logits.dim() == 2
    assert logits.size(0) == 4
    assert logits.size(1) == 1

    print(f"Model forward pass successful. Output shape: {logits.shape}")
    print("Model architecture verification passed.")


def demo_training_pipeline():
    """Demonstrates the full training loop for a single fold."""
    print("\n[Demo] Training Pipeline (Fold 0)")

    # run_fold handles model init, training loop, validation, and prediction
    # debug=True ensures we use the small subset and run quickly
    results = run_fold(fold_index=0, debug=True)

    # Verify results dictionary
    assert "val_loss" in results
    assert "test_preds" in results
    assert "test_ids" in results

    # Verify predictions
    preds = results["test_preds"]
    ids = results["test_ids"]

    assert len(preds) == len(ids)
    # Check probability range
    assert np.all(preds >= 0.0) and np.all(preds <= 1.0)

    print(f"Fold execution complete. Validation Loss: {results['val_loss']:.4f}")

    # Demonstrate creating a submission dataframe (in memory)
    submission = pd.DataFrame({"id": ids, "is_iceberg": preds})
    print("Sample submission rows:")
    print(submission.head())

    print("Training pipeline verification passed.")


if __name__ == "__main__":
    # Execute demos in order
    demo_utils()
    demo_data_processing()
    demo_data_loaders()
    demo_model_architecture()
    demo_training_pipeline()

    print("\nAll demonstrations completed successfully.")
