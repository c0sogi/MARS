import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.features import get_data
from library.dataset import RNADataset
from library.model import StructureShortcutResBiGRU
from library.engine import train_model, predict_and_submit


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("[1/5] Configuring environment for demo run...")

    # Override Config for a fast demonstration
    Config.DEBUG = True
    Config.SUBSET_SIZE = 50  # Use only 50 samples
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_FILE_PATH = os.path.join(
        Config.SUBMISSION_DIR, "demo_submission.csv"
    )

    # Create necessary directories
    Config.setup()

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Debug Mode: {Config.DEBUG}")

    # -------------------------------------------------------------------------
    # 2. Data Processing & Dataset Verification
    # -------------------------------------------------------------------------
    print("\n[2/5] Verifying Data Processing and Dataset...")

    # Test feature extraction logic directly
    # This calls library.features.get_data which handles loading parquet and processing
    raw_data = get_data(split="train", load_cached_data=False)

    # Validate Raw Data Dictionary
    assert "sequence" in raw_data, "Missing 'sequence' in processed data"
    assert "targets" in raw_data, "Missing 'targets' in processed data"
    assert raw_data["sequence"].shape == (
        Config.SUBSET_SIZE,
        107,
    ), f"Expected sequence shape ({Config.SUBSET_SIZE}, 107), got {raw_data['sequence'].shape}"
    # Targets should be (N, 68, 3) corresponding to reactivity, deg_Mg_pH10, deg_Mg_50C
    assert raw_data["targets"].shape == (
        Config.SUBSET_SIZE,
        68,
        3,
    ), f"Expected targets shape ({Config.SUBSET_SIZE}, 68, 3), got {raw_data['targets'].shape}"

    print("    Feature extraction shapes verified.")

    # Test PyTorch Dataset
    dataset = RNADataset(split="train", load_cached_data=True)
    assert len(dataset) == Config.SUBSET_SIZE, "Dataset length mismatch"

    # Fetch one item
    sample = dataset[0]

    # Validate Sample Tensors
    # Sequence: (107,) Long
    assert sample["sequence"].shape == (107,), "Incorrect sequence tensor shape"
    assert sample["sequence"].dtype == torch.long, "Incorrect sequence tensor dtype"

    # Pair Dist: (107, 1) Float
    assert sample["pair_dist"].shape == (107, 1), "Incorrect pair_dist tensor shape"

    # Targets: (68, 3) Float
    assert sample["targets"].shape == (68, 3), "Incorrect targets tensor shape"

    print("    RNADataset item verification passed.")

    # -------------------------------------------------------------------------
    # 3. Model Verification
    # -------------------------------------------------------------------------
    print("\n[3/5] Verifying Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = StructureShortcutResBiGRU().to(device)

    # Create a small dataloader for a forward pass check
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    # Move batch to device
    seq = batch["sequence"].to(device)
    loop = batch["loop_type"].to(device)
    pidx = batch["pair_index"].to(device)
    pdist = batch["pair_dist"].to(device)

    # Forward Pass
    with torch.no_grad():
        output = model(seq, loop, pidx, pdist)

    # Validate Output
    # Model should output predictions for the full sequence length (107) and 3 targets
    expected_shape = (2, 107, 3)
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"

    print(f"    Forward pass successful. Output shape: {output.shape}")

    # -------------------------------------------------------------------------
    # 4. Execution (Training & Inference)
    # -------------------------------------------------------------------------
    print("\n[4/5] Running Training and Inference Pipeline...")

    # Run the training loop (uses library.engine.train_model)
    # This will train for 1 epoch on the subset and save 'best_model.pth'
    train_model()

    # Check if model file exists
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model file was not saved to {Config.MODEL_SAVE_PATH}")
    print("    Training complete. Model saved.")

    # Run inference (uses library.engine.predict_and_submit)
    # This loads the saved model, predicts on test set, and saves CSV
    predict_and_submit()

    if not os.path.exists(Config.SUBMISSION_FILE_PATH):
        raise FileNotFoundError(
            f"Submission file was not created at {Config.SUBMISSION_FILE_PATH}"
        )
    print("    Inference complete. Submission saved.")

    # -------------------------------------------------------------------------
    # 5. Submission Validation
    # -------------------------------------------------------------------------
    print("\n[5/5] Validating Submission File...")

    df_sub = pd.read_csv(Config.SUBMISSION_FILE_PATH)

    # Check columns
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Got {list(df_sub.columns)}"

    # Check rows
    # In DEBUG mode, we used a subset for training, but predict_and_submit loads the TEST set.
    # The get_data('test') function also respects Config.DEBUG and Config.SUBSET_SIZE.
    # So we expect Config.SUBSET_SIZE * 107 rows.
    expected_rows = Config.SUBSET_SIZE * 107
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    # Check values
    # deg_pH10 and deg_50C should be exactly 0.0 as per format_submission logic
    assert (df_sub["deg_pH10"] == 0.0).all(), "deg_pH10 column should be all zeros"
    assert (df_sub["deg_50C"] == 0.0).all(), "deg_50C column should be all zeros"

    print(f"    Submission file valid. Rows: {len(df_sub)}")
    print("\nAll demonstrations passed successfully.")


if __name__ == "__main__":
    main()
