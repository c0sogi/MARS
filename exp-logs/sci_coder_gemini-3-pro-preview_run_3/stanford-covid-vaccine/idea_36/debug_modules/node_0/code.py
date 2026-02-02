import os
import torch
import pandas as pd
import numpy as np
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import set_seed
from library.data import get_loaders
from library.model import DDPNBiGRU
from library.train import (
    train_epoch,
    validate,
    generate_submission_file,
    loss_fn_5_targets,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("============================================================")
    print("       RNA Degradation Prediction - Usage Demonstration     ")
    print("============================================================")

    # ---------------------------------------------------------
    # 1. Configuration Setup
    # ---------------------------------------------------------
    # We override Config parameters to ensure the demo runs quickly and
    # saves outputs to a specific demo directory.
    print("\n[1] Configuring environment...")

    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16  # Reduced batch size for speed
    Config.WORKING_DIR = "./working/demo_run"
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Redirect cache paths to the demo folder to avoid overwriting production caches
    # or using potentially incompatible existing caches.
    Config.TRAIN_CACHE_PATH = os.path.join(Config.WORKING_DIR, "train_data.npz")
    Config.VAL_CACHE_PATH = os.path.join(Config.WORKING_DIR, "val_data.npz")
    Config.TEST_CACHE_PATH = os.path.join(Config.WORKING_DIR, "test_data.npz")

    # Ensure the demo directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set reproducibility seed
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Device: {device}")

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("\n[2] Loading Data...")
    # We set load_cached_data=False to demonstrate the raw processing pipeline from Parquet
    train_loader, val_loader, test_loader = get_loaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=False
    )

    # Inspect a single batch
    batch = next(iter(train_loader))
    X = batch["X"]
    y = batch["y"]
    pair_indices = batch["pair_indices"]

    print(f"    Train Batch X shape: {X.shape}")
    print(f"    Train Batch y shape: {y.shape}")

    # Verification
    # Expected: (Batch, Seq_Len=107, Features=14)
    assert X.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        Config.NUM_NODE_FEATURES,
    ), "X shape mismatch"
    # Expected: (Batch, Seq_Len=107, Targets=5)
    assert y.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        Config.NUM_TARGETS,
    ), "y shape mismatch"
    # Expected: (Batch, Seq_Len=107)
    assert pair_indices.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
    ), "pair_indices shape mismatch"
    print("    Data shapes verified.")

    # ---------------------------------------------------------
    # 3. Model Initialization & Forward Pass
    # ---------------------------------------------------------
    print("\n[3] Initializing Model...")
    model = DDPNBiGRU().to(device)

    # Prepare inputs for device
    X_dev = X.to(device)
    pair_indices_dev = pair_indices.to(device)
    pair_masks_dev = batch["pair_masks"].to(device)
    y_dev = y.to(device)
    target_masks_dev = batch["target_masks"].to(device)

    print("    Running forward pass...")
    preds = model(X_dev, pair_indices_dev, pair_masks_dev)

    print(f"    Output shape: {preds.shape}")
    assert preds.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        Config.NUM_TARGETS,
    ), "Output shape mismatch"

    # ---------------------------------------------------------
    # 4. Loss Calculation
    # ---------------------------------------------------------
    print("\n[4] Calculating Loss...")
    loss = loss_fn_5_targets(preds, y_dev, target_masks_dev)

    print(f"    Batch Loss: {loss.item():.6f}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss must be non-negative"

    # ---------------------------------------------------------
    # 5. Training Loop (1 Epoch)
    # ---------------------------------------------------------
    print("\n[5] Running Training Epoch...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Run a single epoch
    avg_train_loss = train_epoch(model, train_loader, optimizer, device)
    print(f"    Epoch 1 Average Loss: {avg_train_loss:.6f}")

    # ---------------------------------------------------------
    # 6. Validation
    # ---------------------------------------------------------
    print("\n[6] Running Validation...")
    val_mcrmse, val_scores = validate(model, val_loader, device)

    print(f"    Validation MCRMSE: {val_mcrmse:.6f}")
    print(f"    Column Scores: {val_scores}")

    # Check that we have scores for the required columns
    required_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    for col in required_cols:
        assert col in val_scores, f"Missing validation score for {col}"

    # ---------------------------------------------------------
    # 7. Submission Generation
    # ---------------------------------------------------------
    print("\n[7] Generating Submission...")
    generate_submission_file(model, test_loader, device, Config.SUBMISSION_PATH)

    if os.path.exists(Config.SUBMISSION_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"    Submission saved to: {Config.SUBMISSION_PATH}")
        print(f"    Submission shape: {df_sub.shape}")

        # Verify submission dimensions
        # 240 test samples * 107 sequence length = 25680 rows
        expected_rows = 240 * 107
        assert (
            len(df_sub) == expected_rows
        ), f"Expected {expected_rows} rows, got {len(df_sub)}"

        # Verify columns
        expected_cols = [
            "id_seqpos",
            "reactivity",
            "deg_Mg_pH10",
            "deg_pH10",
            "deg_Mg_50C",
            "deg_50C",
        ]
        assert list(df_sub.columns) == expected_cols, "Submission columns mismatch"

        print("    Submission format verified.")
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n============================================================")
    print("       Demonstration Completed Successfully                 ")
    print("============================================================")


if __name__ == "__main__":
    run_demo()
