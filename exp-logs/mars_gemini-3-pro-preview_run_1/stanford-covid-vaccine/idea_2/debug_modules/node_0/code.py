import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- Import Library Components ---
from library.config import Config
from library.utils import process_row
from library.dataset import RNAGraphDataset
from library.model import RNAGNN
from library.trainer import Trainer


def main():
    print("=== Starting RNA Degradation Pipeline Demo ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Demo
    # -------------------------------------------------------------------------
    print("1. Configuring environment for fast demonstration...")

    # Define a separate working directory for this demo
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config attributes to point to the demo directory
    # Note: Since these were defined at class level, we must update dependent paths manually
    Config.WORKING_DIR = DEMO_DIR
    Config.TRAIN_CACHE_PATH = os.path.join(
        DEMO_DIR, "cache", "train_data.npz"
    )  # Extension handled by torch.save usually, but let's stick to .pt logic in utils
    # Actually utils.py uses .pt for torch.save, let's fix the paths to match what utils expects
    # utils.py: torch.save(data_list, cache_path)
    os.makedirs(os.path.join(DEMO_DIR, "cache"), exist_ok=True)
    Config.TRAIN_CACHE_PATH = os.path.join(DEMO_DIR, "cache", "train_data.pt")
    Config.VAL_CACHE_PATH = os.path.join(DEMO_DIR, "cache", "val_data.pt")
    Config.TEST_CACHE_PATH = os.path.join(DEMO_DIR, "cache", "test_data.pt")
    Config.BEST_MODEL_PATH = os.path.join(DEMO_DIR, "best_model.pth")

    # Set DEBUG to True to limit dataset size (utils.load_data handles this)
    Config.DEBUG = True

    # Reduce training parameters for speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    print(f"   Working Directory: {Config.WORKING_DIR}")
    print(f"   Debug Mode: {Config.DEBUG}")
    print(f"   Epochs: {Config.EPOCHS}")

    # -------------------------------------------------------------------------
    # 2. Data Processing Verification (Unit Test style)
    # -------------------------------------------------------------------------
    print("\n2. Verifying Data Processing Logic...")

    # Load one row from metadata to test process_row
    df_train_sample = pd.read_parquet(Config.TRAIN_METADATA_PATH).iloc[0]

    # Process the row
    data_sample = process_row(df_train_sample, is_test=False)

    # Assertions
    # Sequence length is 107
    assert data_sample.x.shape == (
        107,
        3,
    ), f"Expected node features shape (107, 3), got {data_sample.x.shape}"

    # Targets should be (107, 5)
    assert data_sample.y.shape == (
        107,
        5,
    ), f"Expected target shape (107, 5), got {data_sample.y.shape}"

    # Mask should be boolean and length 107
    assert data_sample.mask.shape == (
        107,
    ), f"Expected mask shape (107,), got {data_sample.mask.shape}"

    # Check edge index (2, Num_Edges)
    assert data_sample.edge_index.shape[0] == 2, "Edge index must have 2 rows"

    print("   ✓ Single row processing verified.")

    # -------------------------------------------------------------------------
    # 3. Dataset Loading Verification
    # -------------------------------------------------------------------------
    print("\n3. Verifying Dataset Loading...")

    # Initialize dataset (this will trigger processing and caching)
    train_dataset = RNAGraphDataset(split="train", load_cached_data=False)

    # Since DEBUG=True, load_data limits to 50
    assert (
        len(train_dataset) == 50
    ), f"Expected 50 samples in DEBUG mode, got {len(train_dataset)}"

    # Check first sample
    sample = train_dataset[0]
    assert sample.id is not None, "Sample ID is missing"

    print(f"   ✓ Train dataset loaded with {len(train_dataset)} samples.")

    # -------------------------------------------------------------------------
    # 4. Model Forward Pass Verification
    # -------------------------------------------------------------------------
    print("\n4. Verifying Model Architecture...")

    device = torch.device("cpu")  # Use CPU for simple demo
    model = RNAGNN().to(device)

    # Create a batch of 2 samples
    from torch_geometric.loader import DataLoader

    loader = DataLoader(train_dataset[:2], batch_size=2)
    batch = next(iter(loader)).to(device)

    # Forward pass
    output = model(batch)

    # Expected output shape: (Batch_Size * Seq_Len, Num_Targets)
    # 2 * 107 = 214 nodes
    expected_shape = (2 * 107, 5)
    assert (
        output.shape == expected_shape
    ), f"Expected output shape {expected_shape}, got {output.shape}"

    print("   ✓ Model forward pass successful.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n5. Running Training Loop (Trainer)...")

    # Initialize Trainer
    # Note: Trainer will re-load datasets. Since we already cached them in step 3
    # (via RNAGraphDataset init), it should load fast.
    trainer = Trainer(load_cached_data=True)

    # Override device to CPU if GPU not desired for this tiny demo,
    # but Trainer auto-selects. We let it be.

    # Run Fit
    trainer.fit()

    # Check if model was saved
    assert os.path.exists(Config.BEST_MODEL_PATH), "Best model file was not created."
    print(f"   ✓ Training complete. Model saved to {Config.BEST_MODEL_PATH}")

    # -------------------------------------------------------------------------
    # 6. Inference and Submission Verification
    # -------------------------------------------------------------------------
    print("\n6. Running Inference (Prediction)...")

    # Run Predict
    # This generates ./submission/submission.csv
    trainer.predict()

    submission_path = "./submission/submission.csv"
    assert os.path.exists(submission_path), "Submission file not found."

    # Verify Submission Content
    df_sub = pd.read_csv(submission_path)

    # In DEBUG mode, test set is also limited to 50 samples
    # Total rows should be 50 samples * 107 positions = 5350 rows
    expected_rows = 50 * 107
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows in submission, got {len(df_sub)}"

    # Check columns
    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Expected columns {expected_cols}, got {list(df_sub.columns)}"

    print("   ✓ Submission file generated and verified.")
    print(f"   Submission Shape: {df_sub.shape}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
