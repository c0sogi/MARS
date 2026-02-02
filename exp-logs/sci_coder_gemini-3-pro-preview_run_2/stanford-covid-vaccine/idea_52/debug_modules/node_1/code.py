import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import set_seed, MCRMSELoss
from library.data import preprocess_data, RNADataset
from library.model import SSPFN
from library.train import train_model, generate_submission


def main():
    # ==========================================
    # 1. Setup & Initialization
    # ==========================================
    print("Initializing demonstration...")
    DEMO_DIR = "./working/demo_execution"

    # Clean up previous runs if they exist
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(42)

    # ==========================================
    # 2. Create Mini Datasets for Speed
    # ==========================================
    print("Creating mini datasets from metadata...")
    # Load original metadata
    train_full = pd.read_csv("./metadata/train.csv")
    val_full = pd.read_csv("./metadata/val.csv")
    test_full = pd.read_csv("./metadata/test.csv")

    # Create subsets (16 train, 8 val, 8 test)
    # Sizes chosen to be multiples of batch size (4)
    mini_train = train_full.head(16).copy()
    mini_val = val_full.head(8).copy()
    mini_test = test_full.head(8).copy()

    # Save to demo directory
    mini_train_path = os.path.join(DEMO_DIR, "mini_train.csv")
    mini_val_path = os.path.join(DEMO_DIR, "mini_val.csv")
    mini_test_path = os.path.join(DEMO_DIR, "mini_test.csv")

    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)
    mini_test.to_csv(mini_test_path, index=False)

    # ==========================================
    # 3. Patch Configuration
    # ==========================================
    # We dynamically modify the Config class to use our mini datasets
    # and reduce model complexity for this demonstration.
    print("Patching configuration for rapid execution...")

    # Paths
    Config.WORKING_DIR = DEMO_DIR
    Config.TRAIN_FILE = mini_train_path
    Config.VAL_FILE = mini_val_path
    Config.TEST_FILE = mini_test_path

    # Cache files (to avoid overwriting real cache)
    Config.TRAIN_CACHE = os.path.join(DEMO_DIR, "train_cache.npz")
    Config.VAL_CACHE = os.path.join(DEMO_DIR, "val_cache.npz")
    Config.TEST_CACHE = os.path.join(DEMO_DIR, "test_cache.npz")

    # Outputs
    Config.BEST_MODEL_PATH = os.path.join(DEMO_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "demo_submission.csv")

    # Hyperparameters
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 2
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution
    Config.HIDDEN_DIM = 32  # Reduce dimension for speed
    Config.GROWTH_RATE = 16
    Config.LATENT_DIM = 32

    # ==========================================
    # 4. Verify Data Processing Logic
    # ==========================================
    print("Verifying data processing...")

    # Manually call preprocess_data to check outputs
    feats, p_idx, targets, ids = preprocess_data(
        Config.TRAIN_FILE, Config.TRAIN_CACHE, load_cached_data=False, is_test=False
    )

    # Assertions to ensure data integrity
    assert len(ids) == 16, f"Expected 16 samples, got {len(ids)}"
    # Features: (N, L, 18) -> 18 channels (4 seq + 3 struct + 7 loop + 4 partner)
    assert feats.shape == (16, 107, 18), f"Feature shape mismatch: {feats.shape}"
    # Partner Indices: (N, L)
    assert p_idx.shape == (16, 107), f"Partner indices shape mismatch: {p_idx.shape}"
    # Targets: (N, L, 5)
    assert targets.shape == (16, 107, 5), f"Targets shape mismatch: {targets.shape}"

    # Check partner indices validity (-1 for unpaired, 0-106 for paired)
    assert (p_idx >= -1).all() and (p_idx < 107).all(), "Partner indices out of bounds"

    print("Data processing verified successfully.")

    # ==========================================
    # 5. Verify Model Architecture & Forward Pass
    # ==========================================
    print("Verifying model logic...")

    # Create a DataLoader for a single batch
    dataset = RNADataset(feats, p_idx, targets, ids)
    loader = DataLoader(dataset, batch_size=4)
    b_feats, b_pidx, b_targ = next(iter(loader))

    # Move to configured device
    device = Config.DEVICE
    b_feats = b_feats.to(device)
    b_pidx = b_pidx.to(device)
    b_targ = b_targ.to(device)

    # Instantiate Model
    model = SSPFN().to(device)

    # Test Pass 1: Initial prediction (No feedback)
    pred1 = model(b_feats, b_pidx, feedback_input=None)
    assert pred1.shape == (4, 107, 5), f"Pass 1 output shape mismatch: {pred1.shape}"

    # Test Pass 2: Refined prediction (With feedback)
    pred2 = model(b_feats, b_pidx, feedback_input=pred1)
    assert pred2.shape == (4, 107, 5), f"Pass 2 output shape mismatch: {pred2.shape}"

    # Test Loss Function
    criterion = MCRMSELoss()
    loss = criterion(pred2, b_targ)
    assert loss.item() >= 0, "Loss must be non-negative"

    print("Model architecture and forward pass verified.")

    # ==========================================
    # 6. Run Full Training Loop
    # ==========================================
    print(f"Running training loop for {Config.EPOCHS} epochs...")

    # train_model uses the Config we patched earlier
    train_model(debug=False)

    # Check if model checkpoint was saved
    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError("Best model checkpoint was not created.")

    print("Training loop completed successfully.")

    # ==========================================
    # 7. Generate Submission
    # ==========================================
    print("Generating submission for test set...")

    generate_submission()

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not created.")

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)

    # Check dimensions: 8 test samples * 107 positions = 856 rows
    expected_rows = 8 * 107
    if len(sub_df) != expected_rows:
        raise AssertionError(
            f"Submission rows mismatch. Expected {expected_rows}, got {len(sub_df)}"
        )

    # Check columns
    expected_cols = ["id_seqpos"] + Config.ALL_TARGET_COLS
    if list(sub_df.columns) != expected_cols:
        raise AssertionError(
            f"Submission columns mismatch. Expected {expected_cols}, got {list(sub_df.columns)}"
        )

    print("Submission generation verified.")
    print("\nAll demonstrations passed successfully.")


if __name__ == "__main__":
    main()
