import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders
from library.model import HC_SDRN
from library.loss import MaskedMCRMSELoss
from library.train import train_model, generate_submission


def main():
    print("=== RNA Degradation Prediction Demo ===\n")

    # --------------------------------------------------------------------------
    # 0. Setup Configuration
    # --------------------------------------------------------------------------
    # We override specific Config paths to keep the demo contained in ./working/demo_run
    # and to ensure we don't interfere with other runs.
    DEMO_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Optimize batch size for speed in this demo
    Config.BATCH_SIZE = 8

    # Ensure directories exist
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set reproducibility
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # --------------------------------------------------------------------------
    # 1. Data Loading Verification
    # --------------------------------------------------------------------------
    print("\n[1/5] Loading Data...")

    # get_dataloaders handles loading metadata, processing features, and caching
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=0,  # Use 0 workers for simple demo to avoid multiprocessing overhead
        train_shuffle=True,
    )

    # Verify Train Batch
    try:
        x_batch, p_idx_batch, y_batch = next(iter(train_loader))
        print(f"   Train Batch Loaded:")
        print(
            f"   - Input (X) Shape: {x_batch.shape} (Expected: {Config.BATCH_SIZE}, {Config.SEQ_LEN}, {Config.INPUT_DIM})"
        )
        print(
            f"   - Partner (P) Shape: {p_idx_batch.shape} (Expected: {Config.BATCH_SIZE}, {Config.SEQ_LEN})"
        )
        print(
            f"   - Target (Y) Shape: {y_batch.shape} (Expected: {Config.BATCH_SIZE}, {Config.SEQ_LEN}, 5)"
        )

        # Assertions
        assert x_batch.shape == (
            Config.BATCH_SIZE,
            Config.SEQ_LEN,
            Config.INPUT_DIM,
        ), "Incorrect Input Shape"
        assert p_idx_batch.shape == (
            Config.BATCH_SIZE,
            Config.SEQ_LEN,
        ), "Incorrect Partner Index Shape"
        assert y_batch.shape == (
            Config.BATCH_SIZE,
            Config.SEQ_LEN,
            5,
        ), "Incorrect Target Shape"

    except StopIteration:
        raise Exception("Train loader is empty!")

    # Verify Test Batch (No targets)
    try:
        x_test, p_test = next(iter(test_loader))
        assert x_test.shape == (
            Config.BATCH_SIZE,
            Config.SEQ_LEN,
            Config.INPUT_DIM,
        ), "Incorrect Test Input Shape"
    except StopIteration:
        raise Exception("Test loader is empty!")

    print("   Data loading verified successfully.")

    # --------------------------------------------------------------------------
    # 2. Model Architecture Verification
    # --------------------------------------------------------------------------
    print("\n[2/5] Initializing and Verifying Model...")

    model = HC_SDRN().to(device)

    # Move batch to device
    x_dev = x_batch.to(device)
    p_dev = p_idx_batch.to(device)

    # Perform Forward Pass
    # The model returns a tuple (y1, y2) corresponding to the two passes (initial + recycled)
    y1, y2 = model(x_dev, p_dev)

    print(f"   Model Output Shapes:")
    print(f"   - Pass 1 (y1): {y1.shape}")
    print(f"   - Pass 2 (y2): {y2.shape}")

    assert y1.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        5,
    ), "Output y1 shape mismatch"
    assert y2.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        5,
    ), "Output y2 shape mismatch"

    print("   Model architecture verified successfully.")

    # --------------------------------------------------------------------------
    # 3. Loss Function Verification
    # --------------------------------------------------------------------------
    print("\n[3/5] Verifying Loss Calculation...")

    criterion = MaskedMCRMSELoss()
    y_dev = y_batch.to(device)

    # Calculate loss on the dummy batch
    loss_val = criterion(y2, y_dev)

    print(f"   Calculated Loss: {loss_val.item():.6f}")

    assert not torch.isnan(loss_val), "Loss is NaN"
    assert loss_val.item() >= 0, "Loss is negative"

    print("   Loss function verified successfully.")

    # --------------------------------------------------------------------------
    # 4. Training Loop Demo
    # --------------------------------------------------------------------------
    print("\n[4/5] Running Training Demo (Debug Mode)...")

    # train_model is a high-level function provided in library.train
    # We use debug=True to limit the run to a few batches per epoch for speed
    best_model_path = train_model(epochs=1, debug=True, save_dir=Config.CACHE_DIR)

    assert os.path.exists(best_model_path), "Best model file was not saved."
    print(f"   Training demo complete. Model saved to: {best_model_path}")

    # --------------------------------------------------------------------------
    # 5. Inference and Submission Generation
    # --------------------------------------------------------------------------
    print("\n[5/5] Generating Submission (Debug Mode)...")

    # generate_submission loads the model from best_model_path and predicts on test set
    # debug=True limits inference to a few batches
    generate_submission(
        model_path=best_model_path, output_path=Config.SUBMISSION_PATH, debug=True
    )

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Verify submission format
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"   Submission File Loaded. Shape: {sub_df.shape}")
    print(f"   Columns: {sub_df.columns.tolist()}")

    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    assert list(sub_df.columns) == expected_cols, "Submission columns mismatch"

    # In debug mode (5 batches * 8 samples * 107 seq_len), we expect 4280 rows
    # Note: If test set is smaller than 40 samples, it will be less.
    # Test set is 240 samples, so 40 is fine.
    expected_rows = 5 * Config.BATCH_SIZE * Config.SEQ_LEN
    assert (
        len(sub_df) == expected_rows
    ), f"Expected {expected_rows} rows in debug submission, got {len(sub_df)}"

    print("   Submission generation verified successfully.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
