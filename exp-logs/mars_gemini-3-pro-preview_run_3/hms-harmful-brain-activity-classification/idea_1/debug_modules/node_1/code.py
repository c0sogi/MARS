import os
import shutil
import warnings
import numpy as np
import pandas as pd
import torch

# Import library components
from library.config import Config
from library.utils import seed_everything, AverageMeter
from library.data import get_dataloaders
from library.model import BiGRUModel
from library.train import train_model
from library.inference import generate_submission


def run_demo():
    # ==========================================
    # 1. Setup & Configuration Overrides
    # ==========================================
    print(">>> Setting up configuration for fast demonstration...")

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Patch Config for a fast, lightweight run
    Config.DEBUG = True  # Uses a tiny subset of data (2 * Batch Size)
    Config.EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 4  # Small batch size
    Config.LOAD_CACHED_DATA = False  # Force raw data loading to test processing logic
    Config.NUM_WORKERS = 0  # Use main thread to avoid multiprocessing overhead in demo

    # Clean working directory to ensure we test file creation
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # ==========================================
    # 2. Verify Utilities
    # ==========================================
    print("\n>>> Verifying Utilities...")
    meter = AverageMeter()
    meter.update(val=10, n=2)
    meter.update(val=20, n=2)
    # Expected average: ((10*2) + (20*2)) / 4 = 15
    assert meter.avg == 15, f"AverageMeter logic failed: expected 15, got {meter.avg}"
    print("AverageMeter verified.")

    # ==========================================
    # 3. Verify Data Pipeline
    # ==========================================
    print("\n>>> Verifying Data Pipeline...")
    # This triggers metadata loading and dataset creation
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=Config.LOAD_CACHED_DATA
    )

    # In DEBUG mode, we expect Config.BATCH_SIZE * 2 samples
    # With Batch Size 4, we have 8 samples. Train loader drops last, so 8 // 4 = 2 batches.
    assert len(train_loader) > 0, "Train loader is empty."

    # Fetch a single batch to verify shapes
    x, y = next(iter(train_loader))
    print(f"Batch Input Shape: {x.shape}")  # Expected: (4, 2500, 19)
    print(f"Batch Target Shape: {y.shape}")  # Expected: (4, 6)

    # Assertions
    expected_input_shape = (Config.BATCH_SIZE, Config.SEQ_LENGTH, Config.N_CHANNELS)
    expected_target_shape = (Config.BATCH_SIZE, Config.NUM_CLASSES)

    assert (
        x.shape == expected_input_shape
    ), f"Input shape mismatch. Expected {expected_input_shape}, got {x.shape}"
    assert (
        y.shape == expected_target_shape
    ), f"Target shape mismatch. Expected {expected_target_shape}, got {y.shape}"

    # Verify data integrity
    assert not torch.isnan(x).any(), "Input data contains NaNs."
    assert not torch.isnan(y).any(), "Target data contains NaNs."

    print("Data loading verified.")

    # ==========================================
    # 4. Verify Model Architecture
    # ==========================================
    print("\n>>> Verifying Model Architecture...")
    model = BiGRUModel()
    model.eval()

    # Perform forward pass on CPU
    with torch.no_grad():
        logits = model(x)

    print(f"Logits Shape: {logits.shape}")

    # Assertions
    assert (
        logits.shape == expected_target_shape
    ), f"Model output shape mismatch. Expected {expected_target_shape}, got {logits.shape}"

    print("Model forward pass verified.")

    # ==========================================
    # 5. Verify Training Loop
    # ==========================================
    print("\n>>> Verifying Training Loop...")
    # train_model() handles its own initialization using the global Config
    # It will use the DEBUG settings we applied earlier
    train_model()

    # Verify artifacts were created
    checkpoint_path = os.path.join(Config.WORKING_DIR, "checkpoint.pth")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    assert os.path.exists(
        checkpoint_path
    ), "Checkpoint file (checkpoint.pth) was not created."
    assert os.path.exists(
        best_model_path
    ), "Best model file (best_model.pth) was not created."

    print("Training loop completed and artifacts saved.")

    # ==========================================
    # 6. Verify Inference Pipeline
    # ==========================================
    print("\n>>> Verifying Inference Pipeline...")
    # Generate submission using the model we just trained
    generate_submission(
        device="cpu",  # Use CPU for simple demo
        batch_size=Config.BATCH_SIZE,
        load_cached_data=False,
    )

    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(submission_path), "Submission file was not created."

    # Load and validate submission content
    sub_df = pd.read_csv(submission_path)
    print(f"Submission Shape: {sub_df.shape}")
    print("Submission Head:")
    print(sub_df.head())

    # In DEBUG mode, test set is also truncated to BATCH_SIZE * 2
    expected_rows = Config.BATCH_SIZE * 2
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"

    # Verify probabilities sum to 1.0
    vote_cols = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]

    # Check if columns exist
    for col in vote_cols:
        assert col in sub_df.columns, f"Missing column in submission: {col}"

    # Check sum
    row_sums = sub_df[vote_cols].sum(axis=1)
    # Using a small tolerance for floating point arithmetic
    assert np.allclose(
        row_sums, 1.0, atol=1e-5
    ), "Predicted probabilities do not sum to 1.0."

    print("Inference verified successfully.")
    print("\n>>> Demo Completed Successfully.")


if __name__ == "__main__":
    run_demo()
