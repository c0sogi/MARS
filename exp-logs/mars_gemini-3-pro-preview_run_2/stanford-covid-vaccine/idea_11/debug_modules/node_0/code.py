import sys
import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config
from library.utils import set_seed, MCRMSELoss
from library.data import get_dataloaders, get_structure_pairs, get_one_hot_encoding
from library.model import CascadedDenseNet
from library.train import train_model, generate_submission


def main():
    print("==== Starting Demonstration Script ====")

    # 1. Setup and Configuration Override
    # We override Config parameters to run a fast, isolated demo.
    print("\n[1] Configuring environment for demo...")

    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config paths and parameters
    Config.WORKING_DIR = DEMO_DIR
    Config.TRAIN_CACHE = os.path.join(DEMO_DIR, "data_cache", "train_data.npz")
    Config.VAL_CACHE = os.path.join(DEMO_DIR, "data_cache", "val_data.npz")
    Config.TEST_CACHE = os.path.join(DEMO_DIR, "data_cache", "test_data.npz")
    Config.MODEL_PATH = os.path.join(DEMO_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Reduce compute load for demo
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.DEBUG_SUBSET_SIZE = 10  # Very small subset
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    # Ensure reproducibility
    set_seed(Config.SEED)
    print("Configuration updated.")

    # 2. Demonstrate Data Utilities
    print("\n[2] Demonstrating Data Utilities...")

    # Test get_structure_pairs
    structure = "((..))"
    # Indices: 012345
    # Pairs: 0-5, 1-4. 2,3 are unpaired.
    expected_pairs = np.array([5, 4, 2, 3, 1, 0])
    pairs = get_structure_pairs(structure)
    print(f"Structure: {structure}")
    print(f"Pairs: {pairs}")
    np.testing.assert_array_equal(
        pairs, expected_pairs, err_msg="Structure pairing logic failed"
    )
    print("Structure pairing verification passed.")

    # Test get_one_hot_encoding
    seq = "AGCU"
    struct = "...."
    loop = "EEEE"
    # 4 bases, 3 struct, 7 loop = 14 channels
    encoding = get_one_hot_encoding(seq, struct, loop)
    print(f"Encoding shape: {encoding.shape}")
    assert encoding.shape == (
        4,
        14,
    ), f"Encoding shape mismatch. Expected (4, 14), got {encoding.shape}"

    # Check specific indices to verify mapping
    # 'A' -> index 0
    # '.' -> index 4 (0-3 are seq)
    # 'E' -> index 12 (0-3 seq, 4-6 struct, 7-13 loop. E is 5th in loop map -> 7+5=12)
    assert encoding[0, 0] == 1.0
    assert encoding[0, 4] == 1.0
    assert encoding[0, 12] == 1.0
    print("One-hot encoding verification passed.")

    # 3. Demonstrate Data Loaders
    print("\n[3] Demonstrating Data Loaders...")
    # This will trigger processing and caching since cache files don't exist in DEMO_DIR
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=True, load_cached_data=True
    )

    # Fetch a batch
    batch = next(iter(train_loader))
    inputs = batch["inputs"]
    partners = batch["partner_indices"]
    targets = batch["targets"]

    print(f"Batch Inputs Shape: {inputs.shape}")  # (B, 107, 14)
    print(f"Batch Partners Shape: {partners.shape}")  # (B, 107)
    print(f"Batch Targets Shape: {targets.shape}")  # (B, 68, 5)

    assert inputs.shape == (Config.BATCH_SIZE, Config.SEQ_LENGTH, Config.INPUT_CHANNELS)
    assert partners.shape == (Config.BATCH_SIZE, Config.SEQ_LENGTH)
    # Targets are only available for the first SEQ_SCORED positions
    assert targets.shape == (Config.BATCH_SIZE, Config.SEQ_SCORED, Config.NUM_TARGETS)
    print("Data Loader shapes verified.")

    # 4. Demonstrate Model Architecture
    print("\n[4] Demonstrating Model Architecture...")
    device = Config.DEVICE
    model = CascadedDenseNet().to(device)

    # Move batch to device
    inputs = inputs.to(device)
    partners = partners.to(device)

    # Forward pass
    outputs = model(inputs, partners)
    print(f"Model Output Shape: {outputs.shape}")

    # Model outputs predictions for the full sequence length
    assert outputs.shape == (Config.BATCH_SIZE, Config.SEQ_LENGTH, Config.NUM_TARGETS)
    print("Model forward pass successful.")

    # 5. Demonstrate Loss Function (MCRMSE)
    print("\n[5] Demonstrating MCRMSE Loss...")
    criterion = MCRMSELoss()

    # Create synthetic data to verify masking logic
    # Scored columns are indices 0, 1, 3 (reactivity, deg_Mg_pH10, deg_Mg_50C)
    # We will create an error of 1.0 in a scored column and 100.0 in an unscored column.

    # Batch=1, Seq=68 (to match target length), Cols=5
    pred_t = torch.zeros((1, 68, 5))
    targ_t = torch.zeros((1, 68, 5))

    # Add error to scored column (index 0)
    pred_t[0, :, 0] = 1.0
    targ_t[0, :, 0] = 0.0
    # RMSE for col 0 = sqrt(mean((1-0)^2)) = 1.0

    # Add error to unscored column (index 2 - deg_pH10)
    pred_t[0, :, 2] = 100.0
    targ_t[0, :, 2] = 0.0
    # RMSE for col 2 should be ignored

    # Add error to another scored column (index 1)
    pred_t[0, :, 1] = 2.0
    targ_t[0, :, 1] = 0.0
    # RMSE for col 1 = 2.0

    # Scored col 3 has 0 error.

    # MCRMSE = Mean(RMSE_col0, RMSE_col1, RMSE_col3)
    # MCRMSE = Mean(1.0, 2.0, 0.0) = 1.0

    loss = criterion(pred_t, targ_t)
    print(f"Calculated Loss: {loss.item()}")

    assert (
        abs(loss.item() - 1.0) < 1e-5
    ), f"Loss calculation incorrect. Expected 1.0, got {loss.item()}"
    print("Loss function logic verified.")

    # 6. Demonstrate Training Loop
    print("\n[6] Demonstrating Training Loop (1 Epoch)...")
    # train_model uses the Config we modified earlier
    best_score = train_model(debug=True)
    print(f"Training complete. Best Validation Score: {best_score}")

    assert os.path.exists(
        Config.MODEL_PATH
    ), "Model checkpoint not found after training."
    print("Model checkpoint verified.")

    # 7. Demonstrate Submission Generation
    print("\n[7] Demonstrating Submission Generation...")
    generate_submission(debug=True)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    # Verify submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {sub_df.shape}")
    print("Submission head:")
    print(sub_df.head())

    # Expected rows: DEBUG_SUBSET_SIZE * SEQ_LENGTH
    expected_rows = Config.DEBUG_SUBSET_SIZE * Config.SEQ_LENGTH
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"

    # Expected columns: id_seqpos + 5 targets
    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    assert list(sub_df.columns) == expected_cols, "Submission columns mismatch."

    print("Submission format verified.")
    print("\n==== Demonstration Complete ====")


if __name__ == "__main__":
    main()
