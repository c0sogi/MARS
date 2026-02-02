import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.data import get_dataloaders, CTDataset
from library.model import MAOPDSNet
from library.train import MetricAlignedLLLoss, run_training, train_epoch, validate_epoch


def create_subset_metadata(source_csv, dest_csv, num_patients=4):
    """
    Creates a small subset of the metadata for rapid testing.
    """
    df = pd.read_csv(source_csv)
    # Select a few unique patients
    patients = df["Patient"].unique()[:num_patients]
    subset_df = df[df["Patient"].isin(patients)].copy()

    # Ensure directory exists
    os.makedirs(os.path.dirname(dest_csv), exist_ok=True)
    subset_df.to_csv(dest_csv, index=False)
    print(
        f"Created subset metadata at {dest_csv} with {len(subset_df)} rows ({num_patients} patients)."
    )
    return subset_df


def verify_metric_logic():
    """
    Verifies the metric calculation with known values.
    """
    print("\n--- Verifying Metric Logic ---")

    # Case 1: Perfect prediction (error=0), High confidence (sigma < 70, clipped to 70)
    y_true = np.array([2000])
    y_pred = np.array([2000])
    sigma_pred = np.array([10])  # Should clip to 70

    # Manual Calc:
    # delta = 0
    # sigma_clipped = 70
    # metric = - (sqrt(2)*0)/70 - ln(sqrt(2)*70) = -ln(98.99) approx -4.595

    score = calculate_metric(y_true, y_pred, sigma_pred)
    expected = -np.log(np.sqrt(2) * 70)

    print(f"Case 1 (Perfect): Score={score:.4f}, Expected={expected:.4f}")
    assert np.isclose(
        score, expected, atol=1e-4
    ), "Metric calculation mismatch for perfect prediction."

    # Case 2: Large error (clipped to 1000)
    y_true = np.array([2000])
    y_pred = np.array([4000])  # Error 2000 -> clipped to 1000
    sigma_pred = np.array([100])  # > 70, not clipped

    # Manual Calc:
    # delta = 1000
    # sigma = 100
    # term1 = (1.4142 * 1000) / 100 = 14.142
    # term2 = ln(1.4142 * 100) = ln(141.42) = 4.9517
    # metric = -14.142 - 4.9517 = -19.09

    score = calculate_metric(y_true, y_pred, sigma_pred)
    print(f"Case 2 (Large Error): Score={score:.4f}")
    assert score < -10, "Metric should be significantly negative for large errors."

    print("Metric logic verified.")


def run_demo():
    # 1. Setup
    print("Initializing Demo...")
    seed_everything(42)

    # Define temporary working directory for demo
    demo_dir = "./working/demo_execution_custom"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # 2. Patch Config for Speed
    print("\n--- Patching Configuration ---")
    # We modify the Config class attributes directly.
    # Since the library modules import Config, these changes will propagate
    # as long as they haven't already cached values (which they generally don't for class attrs).

    Config.WORKING_DIR = demo_dir
    Config.CACHE_DIR = os.path.join(demo_dir, "cache")
    Config.CHECKPOINT_DIR = os.path.join(demo_dir, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")
    Config.MODEL_PATH = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    # Create directories
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # Reduce training load
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2  # Small batch size for demo
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Create subset metadata
    subset_train_path = os.path.join(demo_dir, "train_subset.csv")
    subset_val_path = os.path.join(demo_dir, "val_subset.csv")

    # Use existing metadata to create subsets
    create_subset_metadata(
        os.path.join(Config.METADATA_DIR, "train.csv"),
        subset_train_path,
        num_patients=4,
    )
    create_subset_metadata(
        os.path.join(Config.METADATA_DIR, "val.csv"), subset_val_path, num_patients=2
    )

    # Point Config to subsets
    Config.TRAIN_CSV = subset_train_path
    Config.VAL_CSV = subset_val_path
    # Keep Test CSV as is (it's small)

    # 3. Verify Metric
    verify_metric_logic()

    # 4. Data Loading Verification
    print("\n--- Verifying Data Loading ---")
    train_loader, val_loader, test_loader, stats = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    print(f"Train Loader Batches: {len(train_loader)}")
    assert len(train_loader) > 0, "Train loader is empty."

    # Fetch one batch
    batch = next(iter(train_loader))
    images = batch["image"]
    tabular = batch["tabular"]
    targets = batch["target"]

    print(f"Image Shape: {images.shape}")  # Expected: (B, 3, 260, 260)
    print(f"Tabular Shape: {tabular.shape}")  # Expected: (B, 5)
    print(f"Target Shape: {targets.shape}")  # Expected: (B,)

    assert images.shape == (
        Config.BATCH_SIZE,
        Config.NUM_SLICES,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    )
    assert tabular.shape == (Config.BATCH_SIZE, Config.CLINICAL_INPUT_DIM)
    assert targets.shape == (Config.BATCH_SIZE,)

    # Check value ranges (Images should be normalized [0, 1])
    print(f"Image Min: {images.min():.4f}, Max: {images.max():.4f}")
    assert (
        images.min() >= 0.0 and images.max() <= 1.0
    ), "Images not normalized to [0, 1]."

    # 5. Model Verification
    print("\n--- Verifying Model Architecture ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = MAOPDSNet().to(device)

    # Move batch to device
    images = images.to(device)
    tabular = tabular.to(device)
    targets = targets.to(device)

    # Forward Pass
    outputs = model(images, tabular)
    print(f"Model Output Shape: {outputs.shape}")

    assert outputs.shape == (Config.BATCH_SIZE, Config.OUTPUT_DIM)
    assert not torch.isnan(outputs).any(), "Model output contains NaNs."

    # 6. Loss Verification
    print("\n--- Verifying Loss Function ---")
    criterion = MetricAlignedLLLoss()
    loss = criterion(outputs, targets)
    print(f"Loss Value: {loss.item():.4f}")

    assert not torch.isnan(loss), "Loss is NaN."
    assert loss.item() > -100, "Loss seems abnormally low (check formula)."

    # 7. Full Training Cycle Verification
    print("\n--- Verifying Training Loop (1 Epoch) ---")
    # We call run_training directly. It uses Config internally.
    # We already patched Config, so it should run on the subset for 1 epoch.

    try:
        run_training(patience=1)
        print("Training cycle completed successfully.")
    except Exception as e:
        print(f"Training cycle failed: {e}")
        raise e

    # Check if model checkpoint exists
    if os.path.exists(Config.MODEL_PATH):
        print(f"Checkpoint found at {Config.MODEL_PATH}")
    else:
        # It's possible validation metric didn't improve if initialized poorly,
        # but with 1 epoch and -inf start, it should save.
        print(
            "Warning: No checkpoint saved (validation metric might not have improved)."
        )

    # 8. Cleanup
    print("\n--- Cleanup ---")
    # Comment out the next line if you want to inspect the output files
    shutil.rmtree(demo_dir)
    print("Temporary directory removed.")
    print("\nDemo Completed Successfully.")


if __name__ == "__main__":
    run_demo()
