import os
import sys
import shutil
import torch
import numpy as np
import pandas as pd
import warnings

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

# Import library components
from library.config import Config
from library.data import get_dataloaders, RNADataset
from library.model import RHIGFN
from library.loss import mcrmse_loss, GlobalMCRMSE
from library.train import train_one_epoch, validate, generate_submission, set_seed


def run_demo():
    print("Initializing Demo Script...")

    # 1. Setup Configuration for Demo
    # We modify the Config class attributes directly to suit a fast demo run.

    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    print(f"Setting up demo configuration in {demo_dir}...")

    # Override Config paths to isolate demo artifacts
    Config.WORKING_DIR = demo_dir
    Config.TRAIN_CACHE = os.path.join(demo_dir, "demo_train.npz")
    Config.VAL_CACHE = os.path.join(demo_dir, "demo_val.npz")
    Config.TEST_CACHE = os.path.join(demo_dir, "demo_test.npz")
    Config.BEST_MODEL_PATH = os.path.join(demo_dir, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "demo_submission.csv")

    # Speed optimizations for demonstration
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 20  # Very small subset for speed
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny demo
    Config.DEVICE = "cpu"  # Force CPU for simple verification

    # Reproducibility
    set_seed(Config.SEED)

    # 2. Data Loading Demonstration
    print("\n--- Testing Data Loading ---")
    # This will trigger processing of the small debug subset and caching it
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        debug=Config.DEBUG
    )

    print(f"Train Loader batches: {len(train_loader)}")
    print(f"Val Loader batches: {len(val_loader)}")
    print(f"Test Loader batches: {len(test_loader)}")

    # Verification: Check batch structure
    x, p_idx, y = next(iter(train_loader))

    # Expected Shapes:
    # x: (Batch, SeqLen=107, Channels=18)
    # p_idx: (Batch, SeqLen=107)
    # y: (Batch, SeqLen=107, Targets=5)

    print(f"Input Batch Shape: {x.shape}")
    print(f"Partner Indices Shape: {p_idx.shape}")
    print(f"Target Batch Shape: {y.shape}")

    assert x.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        Config.IN_CHANNELS,
    ), f"Incorrect input shape: {x.shape}"
    assert p_idx.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
    ), f"Incorrect partner index shape: {p_idx.shape}"
    assert y.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        Config.NUM_TARGETS,
    ), f"Incorrect target shape: {y.shape}"

    print("Data loading verification passed.")

    # 3. Model Instantiation & Forward Pass
    print("\n--- Testing Model Architecture ---")
    device = torch.device(Config.DEVICE)

    model = RHIGFN().to(device)
    model.eval()

    # Pass 1: No Feedback (Initial prediction)
    print("Running Forward Pass 1 (No Feedback)...")
    with torch.no_grad():
        pred_1 = model(x.to(device), p_idx.to(device), feedback=None)

    assert pred_1.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        Config.NUM_TARGETS,
    ), f"Pass 1 output shape mismatch: {pred_1.shape}"

    # Pass 2: With Feedback (Refinement)
    print("Running Forward Pass 2 (With Feedback)...")
    with torch.no_grad():
        # Using pred_1 as feedback input
        pred_2 = model(x.to(device), p_idx.to(device), feedback=pred_1)

    assert pred_2.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        Config.NUM_TARGETS,
    ), f"Pass 2 output shape mismatch: {pred_2.shape}"

    print("Model forward pass verification passed.")

    # 4. Loss Function Test
    print("\n--- Testing Loss Functions ---")
    # Move targets to device
    y_dev = y.to(device)

    # Calculate MCRMSE Loss (Scalar)
    loss_val = mcrmse_loss(pred_2, y_dev)
    print(f"Calculated MCRMSE Loss: {loss_val.item():.6f}")

    assert not torch.isnan(loss_val), "Loss is NaN"
    assert loss_val.item() >= 0, "Loss must be non-negative"

    # Test Global Metric Accumulator (Used for Validation)
    print("Testing GlobalMCRMSE Accumulator...")
    metric = GlobalMCRMSE()
    metric.update(pred_2, y_dev)
    # Update with a second batch (simulated by reusing same batch)
    metric.update(pred_2, y_dev)

    global_score = metric.compute()
    print(f"Global Metric Score: {global_score:.6f}")
    assert global_score > 0, "Global score should be positive"

    print("Loss function verification passed.")

    # 5. Training Loop Integration
    print("\n--- Testing Training Step ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Run one epoch of training using the library function
    # train_one_epoch(model, loader, optimizer, device, epoch)
    avg_loss = train_one_epoch(model, train_loader, optimizer, device, epoch=0)
    print(f"Training Epoch 1 Avg Loss: {avg_loss:.6f}")
    assert avg_loss > 0, "Training loss should be positive"

    # 6. Validation Integration
    print("\n--- Testing Validation Step ---")
    # validate(model, loader, device)
    val_score = validate(model, val_loader, device)
    print(f"Validation Score: {val_score:.6f}")
    assert val_score >= 0, "Validation score should be non-negative"

    # 7. Submission Generation
    print("\n--- Testing Submission Generation ---")
    # Save the current model state to the path expected by generate_submission (via Config)
    torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    # Generate submission using the library function
    generate_submission(model, test_loader, test_ids, device, Config.SUBMISSION_PATH)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission DataFrame Shape: {df_sub.shape}")

    # Expected rows: Num_Test_Samples (20 in debug) * Seq_Length (107)
    # Note: Test set size is min(240, DEBUG_SUBSET_SIZE)
    expected_samples = min(240, Config.DEBUG_SUBSET_SIZE)
    expected_rows = expected_samples * Config.SEQ_LENGTH

    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows in submission, got {len(df_sub)}"

    # Check columns
    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    assert list(df_sub.columns) == expected_cols, "Submission columns mismatch"

    print("Submission verification passed.")

    print("\nAll demo steps completed successfully.")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    try:
        run_demo()
    except AssertionError as e:
        print(f"ASSERTION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"EXECUTION FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
