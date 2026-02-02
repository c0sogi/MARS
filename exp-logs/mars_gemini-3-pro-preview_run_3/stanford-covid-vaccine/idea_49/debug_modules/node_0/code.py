import os
import sys
import torch
import pandas as pd
import numpy as np

# 1. Import and Patch Configuration
# We patch the Config class before importing other modules to ensure
# they use the modified settings (e.g., smaller model, new paths).
from library.config import Config

# Define a separate working directory for the demo to avoid overwriting real experiments
DEMO_DIR = "./working/demo_execution"
os.makedirs(DEMO_DIR, exist_ok=True)

# Patch Paths
Config.WORKING_DIR = DEMO_DIR
Config.TRAIN_CACHE = os.path.join(DEMO_DIR, "train_cache.npy")
Config.VAL_CACHE = os.path.join(DEMO_DIR, "val_cache.npy")
Config.TEST_CACHE = os.path.join(DEMO_DIR, "test_cache.npy")
Config.MODEL_PATH = os.path.join(DEMO_DIR, "demo_model.pth")
Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "demo_submission.csv")

# Patch Hyperparameters for Speed
Config.DEBUG = True
Config.DEBUG_SUBSET_SIZE = 32  # Use only 32 samples for training
Config.BATCH_SIZE = 8
Config.MAX_EPOCHS = 2  # Run only 2 epochs
Config.HIDDEN_DIM = 64  # Reduce model capacity for speed
Config.N_LAYERS = 2  # Reduce depth
Config.CONV_FILTERS = 32  # Reduce convolution filters
Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

# 2. Import Library Modules
# Now that Config is patched, we import the rest of the library
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import DeepStabilizedBiGRU
from library.loss import MCRMSELoss
from library.train import train_and_predict

if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running demo on device: {device}")
    print(f"Working directory: {Config.WORKING_DIR}")

    # ==========================================
    # Step 1: Data Pipeline Verification
    # ==========================================
    print("\n[1/5] Verifying Data Pipeline...")

    # Initialize dataloaders
    # This will trigger preprocessing and caching in the demo directory
    train_loader, val_loader, test_loader = get_dataloaders(debug=True)

    # Fetch one batch
    batch = next(iter(train_loader))
    x, bppm, y, ids = batch

    # Move to device for consistency checks
    x = x.to(device)
    bppm = bppm.to(device)
    y = y.to(device)

    print(
        f"  Batch Shapes -> Inputs: {x.shape}, BPPM: {bppm.shape}, Targets: {y.shape}"
    )

    # Assertions
    # x: (Batch, Seq_Len, Channels)
    assert x.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.INPUT_CHANNELS,
    ), "Input shape mismatch"
    # bppm: (Batch, Seq_Len)
    assert bppm.shape == (Config.BATCH_SIZE, Config.SEQ_LEN), "BPPM shape mismatch"
    # y: (Batch, Seq_Scored, Num_Targets) - Note: Loader provides scored length for targets
    assert y.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_SCORED,
        Config.NUM_TARGETS,
    ), "Target shape mismatch"

    print("  Data Pipeline Verified.")

    # ==========================================
    # Step 2: Model Verification
    # ==========================================
    print("\n[2/5] Verifying Model Architecture...")

    # Instantiate model
    model = DeepStabilizedBiGRU().to(device)

    # Forward pass
    preds = model(x, bppm)

    print(f"  Prediction Shape: {preds.shape}")

    # Assertions
    # Output should be (Batch, Seq_Len, Num_Targets)
    # Note: Model outputs full sequence length (107), not just scored length
    assert preds.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.NUM_TARGETS,
    ), "Model output shape mismatch"

    print("  Model Architecture Verified.")

    # ==========================================
    # Step 3: Loss Function Verification
    # ==========================================
    print("\n[3/5] Verifying MCRMSE Loss...")

    criterion = MCRMSELoss()

    # Calculate loss
    loss = criterion(preds, y)

    print(f"  Calculated Loss: {loss.item():.6f}")

    # Assertions
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss is negative"
    # Since model is untrained, loss should be reasonably high (not near zero)
    assert loss.item() > 0.01, "Loss is suspiciously low for untrained model"

    print("  Loss Function Verified.")

    # ==========================================
    # Step 4: Full Training Loop Execution
    # ==========================================
    print("\n[4/5] Executing Training Loop (Demo)...")

    # Run the high-level training function
    # This handles training, validation, checkpointing, and inference
    train_and_predict(debug=True, epochs=Config.MAX_EPOCHS)

    print("  Training Loop Completed.")

    # ==========================================
    # Step 5: Submission Output Verification
    # ==========================================
    print("\n[5/5] Verifying Submission Output...")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file was not created at {Config.SUBMISSION_PATH}"
        )

    # Load submission
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)

    print(f"  Submission Shape: {sub_df.shape}")
    print(f"  First 3 rows:\n{sub_df.head(3)}")

    # Assertions
    # Test set has 240 samples. Predictions are for full length 107.
    # Total rows should be 240 * 107 = 25680
    expected_rows = 240 * 107
    assert (
        len(sub_df) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(sub_df)}"

    # Check columns
    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    assert list(sub_df.columns) == expected_cols, "Submission columns mismatch"

    print("  Submission Output Verified.")

    print("\nAll demo checks passed successfully!")
