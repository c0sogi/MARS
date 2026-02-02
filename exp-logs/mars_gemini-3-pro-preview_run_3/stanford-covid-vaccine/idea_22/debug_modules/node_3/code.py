import os
import torch
import numpy as np
import pandas as pd
import warnings

# Import library components
from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders, RNADataset
from library.model import NonLinearChannelGatedBiGRU
from library.loss import MCRMSELoss
from library.train import run_training

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    print("==== Starting Library Demonstration ====")

    # 1. Setup Configuration for Debugging
    # We override Config class attributes to ensure all components use our temporary demo directory
    # and fast settings.
    DEMO_DIR = "./working/demo_execution"
    os.makedirs(DEMO_DIR, exist_ok=True)

    print(f"Setting up configuration in {DEMO_DIR}...")

    # Override paths to avoid conflicts with main training artifacts
    Config.WORKING_DIR = DEMO_DIR
    Config.TRAIN_CACHE = os.path.join(DEMO_DIR, "train_cache.npy")
    Config.VAL_CACHE = os.path.join(DEMO_DIR, "val_cache.npy")
    Config.TEST_CACHE = os.path.join(DEMO_DIR, "test_cache.npy")
    Config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Override hyperparameters for speed
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.NUM_WORKERS = (
        0  # Use 0 workers for simple debugging to avoid multiprocessing overhead
    )

    # Set reproducibility
    set_seed(Config.SEED)

    # 2. Verify Data Loading
    print("\n[1/5] Verifying Data Loading...")
    # We use debug=True to load only 100 samples per split
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=True,
        load_cached_data=False,  # Force processing from parquet
        batch_size=Config.BATCH_SIZE,
    )

    # Fetch a single batch to inspect
    batch = next(iter(train_loader))
    inputs = batch["inputs"]
    pair_indices = batch["pair_indices"]
    targets = batch["targets"]
    ids = batch["id"]

    print(f"  Batch Shapes:")
    print(f"  - Inputs: {inputs.shape} (Expected: {Config.BATCH_SIZE}, 107, 14)")
    print(f"  - Pairs:  {pair_indices.shape} (Expected: {Config.BATCH_SIZE}, 107)")
    print(f"  - Targets: {targets.shape} (Expected: {Config.BATCH_SIZE}, 68, 5)")

    # Assertions to ensure data integrity
    assert inputs.shape == (Config.BATCH_SIZE, 107, 14), "Input shape mismatch"
    assert pair_indices.shape == (Config.BATCH_SIZE, 107), "Pair indices shape mismatch"
    assert targets.shape == (Config.BATCH_SIZE, 68, 5), "Target shape mismatch"
    print("  Data loading verification passed.")

    # 3. Verify Model Architecture
    print("\n[2/5] Verifying Model Architecture...")
    device = "cpu"  # Use CPU for this quick check
    model = NonLinearChannelGatedBiGRU().to(device)

    # Perform a forward pass
    outputs = model(inputs.to(device), pair_indices.to(device))

    print(f"  Model Output Shape: {outputs.shape}")

    # Output should be (Batch, Seq_Len, 5)
    assert outputs.shape == (Config.BATCH_SIZE, 107, 5), "Model output shape mismatch"
    print("  Model forward pass verification passed.")

    # 4. Verify Loss Function
    print("\n[3/5] Verifying Loss Function...")
    criterion = MCRMSELoss()

    # Calculate loss
    outputs_scored = outputs[:, : Config.PRED_LEN, :]
    loss = criterion(outputs_scored, targets.to(device))
    print(f"  Calculated MCRMSE Loss: {loss.item():.6f}")

    # Basic sanity checks
    assert loss.item() >= 0, "Loss must be non-negative"
    assert not torch.isnan(loss), "Loss must not be NaN"
    print("  Loss function verification passed.")

    # 5. Run Full Training Pipeline (Debug Mode)
    print("\n[4/5] Running Training Pipeline (Debug Mode)...")
    # This will train for 1 epoch on the 100-sample subset and generate a submission
    run_training(debug=True, epochs=1, batch_size=Config.BATCH_SIZE)

    # 6. Verify Submission Output
    print("\n[5/5] Verifying Submission File...")
    if os.path.exists(Config.SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"  Submission File Loaded. Shape: {sub_df.shape}")

        # In debug mode, test set has 100 samples.
        # Submission rows = 100 samples * 107 positions = 10700 rows.
        expected_rows = 100 * 107
        assert (
            len(sub_df) == expected_rows
        ), f"Expected {expected_rows} rows in debug submission, found {len(sub_df)}"

        # Check columns
        expected_cols = [
            "id_seqpos",
            "reactivity",
            "deg_Mg_pH10",
            "deg_pH10",
            "deg_Mg_50C",
            "deg_50C",
        ]
        assert list(sub_df.columns) == expected_cols, "Submission columns mismatch"

        print("  Submission verification passed.")
    else:
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    print("\n==== Library Demonstration Completed Successfully ====")
