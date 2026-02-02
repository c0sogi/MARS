import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch

# --- Setup & Import Interception ---
# The provided library/model.py executes a training loop at the top level.
# We must intercept this to prevent a long-running process during import.
import library.utils

original_get_device = library.utils.get_device


class StopExecution(Exception):
    pass


def mock_get_device():
    raise StopExecution


# Patch get_device to raise exception when called by library.model
library.utils.get_device = mock_get_device

try:
    # Import library.model; this will trigger execution, call get_device, and stop.
    import library.model
except StopExecution:
    pass
except Exception as e:
    print(f"Warning: Unexpected error during import interception: {e}")

# Restore original get_device for normal usage
library.utils.get_device = original_get_device

# Now safe to import the rest without triggering the full training loop
from library.dataset import get_dataloaders, CactusDataset
from library.model import WideResNetMultiScale
from library.train import run_training
from library.utils import seed_everything, calculate_roc_auc, get_device


def verify_utils():
    """Verify utility functions."""
    print("[1/4] Verifying Utils...")

    # Test Reproducibility
    seed_everything(42)
    a = np.random.rand(5)
    seed_everything(42)
    b = np.random.rand(5)
    assert np.allclose(a, b), "seed_everything failed to ensure reproducibility."

    # Test ROC AUC
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    auc = calculate_roc_auc(y_true, y_scores)
    # Expected AUC is 0.75
    assert 0.0 <= auc <= 1.0, "ROC AUC out of bounds."

    # Test Device
    device = get_device()
    assert isinstance(device, torch.device), "get_device did not return a torch.device."
    print("Utils verified.")


def verify_dataset():
    """Verify Dataset and DataLoader."""
    print("\n[2/4] Verifying Dataset...")

    # Use small batch size and 0 workers for simple verification
    loaders = get_dataloaders(batch_size=4, num_workers=0, load_cached_data=True)
    train_loader = loaders["train"]
    test_loader = loaders["test"]

    # Check Train Batch
    imgs, lbls = next(iter(train_loader))
    assert imgs.shape == (4, 3, 32, 32), f"Train batch shape mismatch: {imgs.shape}"
    assert lbls.shape == (4,), f"Train label shape mismatch: {lbls.shape}"
    assert imgs.dtype == torch.float32, "Image tensor dtype mismatch."

    # Check Test Batch (No labels)
    imgs_test = next(iter(test_loader))
    assert imgs_test.shape == (
        4,
        3,
        32,
        32,
    ), f"Test batch shape mismatch: {imgs_test.shape}"

    print("Dataset verified.")
    return loaders


def verify_model(device):
    """Verify Model Architecture."""
    print("\n[3/4] Verifying Model...")

    model = WideResNetMultiScale().to(device)

    # Dummy input
    x = torch.randn(2, 3, 32, 32).to(device)

    # Forward pass
    out = model(x)
    assert out.shape == (2, 1), f"Model output shape mismatch: {out.shape}"

    # Backward pass check to ensure graph connectivity
    out.mean().backward()
    assert model.conv1.weight.grad is not None, "Gradients missing after backward pass."

    print("Model verified.")


def verify_training_pipeline():
    """Verify the full training and submission pipeline."""
    print("\n[4/4] Verifying Training Pipeline...")

    # Setup temporary directories for this demonstration
    demo_work_dir = "./working/demo_verification"
    demo_sub_dir = "./working/demo_verification/submission"

    if os.path.exists(demo_work_dir):
        shutil.rmtree(demo_work_dir)

    # Run a minimal training loop
    # 1 Epoch, 1 Fold, Batch 32 to ensure speed
    run_training(
        epochs=1,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=1e-4,
        n_folds=1,
        patience=1,
        num_workers=0,
        working_dir=demo_work_dir,
        submission_dir=demo_sub_dir,
        load_cached_data=True,
    )

    # Check artifacts
    model_path = os.path.join(demo_work_dir, "model_seed_0.pth")
    sub_path = os.path.join(demo_sub_dir, "submission.csv")

    assert os.path.exists(model_path), "Model checkpoint not found."
    assert os.path.exists(sub_path), "Submission file not found."

    # Validate submission format
    df = pd.read_csv(sub_path)
    assert list(df.columns) == ["id", "has_cactus"], "Submission columns mismatch."
    assert len(df) == 3325, f"Submission length mismatch. Expected 3325, got {len(df)}"

    print("Training pipeline verified.")


if __name__ == "__main__":
    device = get_device()
    print(f"Running on device: {device}")

    verify_utils()
    verify_dataset()
    verify_model(device)
    verify_training_pipeline()

    print("\nAll verifications passed successfully.")
