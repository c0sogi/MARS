import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
import shutil

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Import from the provided library files
from library.config import Config
from library.data import get_dataloaders
from library.model import RNAModel
from library.train import train_model, generate_submission
from library.utils import set_seed


def run_demo():
    print("Initializing Demo Configuration...")

    # 1. Configuration Override for Speed and Demo purposes
    class DemoConfig(Config):
        # Use a separate working directory for the demo
        WORKING_DIR = "./working/demo_run"
        CACHE_DIR = os.path.join(WORKING_DIR, "cache")
        MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
        SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

        # Reduce dataset size and training duration
        DEBUG = True
        DEBUG_SAMPLES = 20  # Only use 20 samples for verification
        EPOCHS = 2  # Run only 2 epochs
        BATCH_SIZE = 4  # Small batch size
        NUM_WORKERS = 0  # Disable multiprocessing for simple demo execution

        # Ensure reproducibility
        SEED = 42

    # Create directories
    DemoConfig.create_dirs()
    set_seed(DemoConfig.SEED)

    print(f"Demo Working Directory: {DemoConfig.WORKING_DIR}")

    # --------------------------------------------------------------------------
    # 2. Data Pipeline Verification
    # --------------------------------------------------------------------------
    print("\n[Step 1] Verifying Data Pipeline...")

    # Generate DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        DemoConfig, load_cached_data=False
    )

    # Fetch one batch to verify structure
    batch = next(iter(train_loader))

    # Extract components
    seq = batch["sequence"]
    loop = batch["loop"]
    dist = batch["distance"]
    target = batch["target"]
    ids = batch["id"]

    print(f"  Batch Size: {DemoConfig.BATCH_SIZE}")
    print(f"  Sequence Shape: {seq.shape}")
    print(f"  Target Shape: {target.shape}")

    # Assertions
    # Sequence length should be 107
    assert seq.shape == (
        DemoConfig.BATCH_SIZE,
        DemoConfig.SEQ_LEN,
    ), f"Expected sequence shape ({DemoConfig.BATCH_SIZE}, 107), got {seq.shape}"

    # Target length should be 68 (scored positions) and 5 channels
    assert target.shape == (
        DemoConfig.BATCH_SIZE,
        DemoConfig.PRED_LEN,
        DemoConfig.NUM_TARGETS,
    ), f"Expected target shape ({DemoConfig.BATCH_SIZE}, 68, 5), got {target.shape}"

    # Distance matrix (encoded as 1D relative distances here) should be length 107
    assert dist.shape == (
        DemoConfig.BATCH_SIZE,
        DemoConfig.SEQ_LEN,
    ), f"Expected distance shape ({DemoConfig.BATCH_SIZE}, 107), got {dist.shape}"

    print("  Data Pipeline verification passed.")

    # --------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # --------------------------------------------------------------------------
    print("\n[Step 2] Verifying Model Architecture...")

    device = torch.device("cpu")  # Use CPU for simple logic check
    model = RNAModel(DemoConfig).to(device)

    # Perform forward pass
    with torch.no_grad():
        outputs = model(seq.to(device), loop.to(device), dist.to(device))

    print(f"  Output Shape: {outputs.shape}")

    # Assertions
    # Model should output predictions for the full sequence length (107)
    # Even though we only score the first 68, the architecture usually processes the whole seq.
    assert outputs.shape == (
        DemoConfig.BATCH_SIZE,
        DemoConfig.SEQ_LEN,
        DemoConfig.NUM_TARGETS,
    ), f"Expected output shape ({DemoConfig.BATCH_SIZE}, 107, 5), got {outputs.shape}"

    print("  Model Architecture verification passed.")

    # --------------------------------------------------------------------------
    # 4. Training Loop Execution
    # --------------------------------------------------------------------------
    print("\n[Step 3] Executing Training Loop (Demo)...")

    # Run the training function provided in library.train
    # This will train for 2 epochs on the debug subset and save the model
    train_model(DemoConfig)

    # Verify model file was created
    if not os.path.exists(DemoConfig.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model file not found at {DemoConfig.MODEL_SAVE_PATH} after training."
        )

    print("  Training execution successful.")

    # --------------------------------------------------------------------------
    # 5. Inference and Submission Generation
    # --------------------------------------------------------------------------
    print("\n[Step 4] Generating Submission...")

    # Run the submission generation function
    generate_submission(DemoConfig)

    # Verify submission file
    if not os.path.exists(DemoConfig.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {DemoConfig.SUBMISSION_PATH}"
        )

    # Load submission to check format
    df_sub = pd.read_csv(DemoConfig.SUBMISSION_PATH)
    print(f"  Submission Shape: {df_sub.shape}")
    print(f"  Submission Columns: {list(df_sub.columns)}")

    # Expected rows: Number of test samples * Sequence Length
    # In DEBUG mode, we limited the dataset.
    # The `process_data` function in `library.data` applies DEBUG slicing to the dataframe.
    # So the test set size will be DEBUG_SAMPLES (20).
    expected_rows = DemoConfig.DEBUG_SAMPLES * DemoConfig.SEQ_LEN
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows in submission, got {len(df_sub)}"

    # Expected columns
    expected_cols = ["id_seqpos"] + DemoConfig.TARGET_COLS
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns do not match requirements. Got {list(df_sub.columns)}"

    # Check content format (id_seqpos)
    sample_id_seqpos = df_sub.iloc[0]["id_seqpos"]
    assert (
        "_0" in sample_id_seqpos or "_1" in sample_id_seqpos
    ), f"id_seqpos format seems incorrect: {sample_id_seqpos}"

    print("  Submission verification passed.")
    print("\nAll demo tasks completed successfully!")


if __name__ == "__main__":
    run_demo()
