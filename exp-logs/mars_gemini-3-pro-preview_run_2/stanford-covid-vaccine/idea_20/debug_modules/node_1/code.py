import os
import sys
import shutil
import warnings
import pandas as pd
import torch
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore")

# Import library modules
from library.config import Config
from library.utils import seed_everything, GlobalMetricTracker
from library.loss import MaskedMCRMSELoss
from library.data import get_loaders
from library.model import ScalePartitionedDenseNet
from library.train import run_training


def create_demo_subsets(n_samples=20):
    """
    Creates small subsets of the metadata CSVs to speed up the demo.
    """
    print(f"\n[Demo] Creating data subsets (n={n_samples})...")

    demo_meta_dir = "./working/demo_metadata"
    os.makedirs(demo_meta_dir, exist_ok=True)

    # Source paths
    src_train = "./metadata/train.csv"
    src_val = "./metadata/val.csv"
    src_test = "./metadata/test.csv"

    # Target paths
    dst_train = os.path.join(demo_meta_dir, "train_subset.csv")
    dst_val = os.path.join(demo_meta_dir, "val_subset.csv")
    dst_test = os.path.join(demo_meta_dir, "test_subset.csv")

    # Read and save subsets
    pd.read_csv(src_train).head(n_samples).to_csv(dst_train, index=False)
    pd.read_csv(src_val).head(n_samples).to_csv(dst_val, index=False)
    pd.read_csv(src_test).head(n_samples).to_csv(dst_test, index=False)

    return dst_train, dst_val, dst_test


def configure_demo_paths(train_path, val_path, test_path):
    """
    Overrides Config paths to use the demo subsets and a separate working dir.
    """
    print("[Demo] Configuring library.Config for demo execution...")

    # Override Input Paths
    Config.TRAIN_CSV = train_path
    Config.VAL_CSV = val_path
    Config.TEST_CSV = test_path

    # Override Output/Working Paths
    Config.IDEA_NAME = "demo_execution"
    Config.IDEA_DIR = os.path.join(Config.WORKING_DIR, Config.IDEA_NAME)
    Config.SUBMISSION_PATH = os.path.join(Config.IDEA_DIR, "submission.csv")

    # Override Cache Paths to ensure we generate new .npz files for the subsets
    # We use a unique version suffix for the demo
    version = "demo_v1"
    Config.CACHE_VERSION = version
    Config.TRAIN_DATA_PATH = os.path.join(Config.IDEA_DIR, f"train_data_{version}.npz")
    Config.VAL_DATA_PATH = os.path.join(Config.IDEA_DIR, f"val_data_{version}.npz")
    Config.TEST_DATA_PATH = os.path.join(Config.IDEA_DIR, f"test_data_{version}.npz")

    # Ensure the directory exists
    Config.setup()


def verify_components():
    """
    Tests individual components: Data, Model, Loss, Metric.
    """
    print("\n[Demo] Verifying components...")

    # 1. Verify Data Loaders
    print("  -> Verifying Data Loaders...")
    # Use a small batch size for verification
    Config.BATCH_SIZE = 4
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=False)

    batch = next(iter(train_loader))
    inputs, partner_indices, targets = batch

    # Check shapes
    # Inputs: (Batch, SeqLen, Channels=18)
    assert inputs.ndim == 3
    assert inputs.shape[1] == 107
    assert inputs.shape[2] == 18

    # Partner Indices: (Batch, SeqLen)
    assert partner_indices.ndim == 2
    assert partner_indices.shape[1] == 107

    # Targets: (Batch, SeqLen, 5)
    assert targets.ndim == 3
    assert targets.shape[2] == 5

    print("     Data Loader shapes correct.")

    # 2. Verify Model
    print("  -> Verifying ScalePartitionedDenseNet...")
    device = "cpu"  # Run verification on CPU
    model = ScalePartitionedDenseNet().to(device)

    # Forward pass
    preds = model(inputs.to(device), partner_indices.to(device))

    # Check Output Shape: (Batch, SeqLen, 5)
    assert preds.shape == (
        inputs.shape[0],
        107,
        5,
    ), f"Expected output shape {(inputs.shape[0], 107, 5)}, got {preds.shape}"

    print("     Model forward pass successful.")

    # 3. Verify Loss
    print("  -> Verifying MaskedMCRMSELoss...")
    loss_fn = MaskedMCRMSELoss()

    # Slice to scored sequence length as done in training loop
    seq_scored = Config.SEQ_SCORED
    preds_sliced = preds[:, :seq_scored, :]
    targets_sliced = targets[:, :seq_scored, :].to(device)

    loss = loss_fn(preds_sliced, targets_sliced)

    assert isinstance(loss.item(), float)
    assert loss.item() >= 0
    print(f"     Loss calculation successful. Value: {loss.item():.4f}")

    # 4. Verify Metric Tracker
    print("  -> Verifying GlobalMetricTracker...")
    tracker = GlobalMetricTracker()

    # Tracker expects numpy arrays
    p_np = preds_sliced.detach().numpy()
    t_np = targets_sliced.detach().numpy()

    # Simulate update
    tracker.update(p_np, t_np)
    metric = tracker.compute()

    assert metric >= 0
    print(f"     Metric computation successful. Value: {metric:.4f}")


def run_pipeline_demo():
    """
    Runs the full training and inference pipeline using the subset data.
    """
    print("\n[Demo] Running full training pipeline (2 epochs)...")

    # Run training with minimal epochs and batch size
    run_training(epochs=2, batch_size=4)

    # Verify Submission
    submission_path = Config.SUBMISSION_PATH
    if os.path.exists(submission_path):
        df = pd.read_csv(submission_path)
        print(f"  -> Submission file generated at {submission_path}")
        print(f"  -> Shape: {df.shape}")

        # Expected rows: n_test_samples (20) * seq_len (107) = 2140
        n_test_samples = 20
        expected_rows = n_test_samples * 107
        assert (
            len(df) == expected_rows
        ), f"Expected {expected_rows} rows in submission, got {len(df)}"

        # Check columns
        expected_cols = ["id_seqpos"] + Config.TARGET_COLS
        assert (
            list(df.columns) == expected_cols
        ), f"Expected columns {expected_cols}, got {list(df.columns)}"

        print("  -> Submission content verified.")
    else:
        raise FileNotFoundError("Submission file was not created.")


if __name__ == "__main__":
    # Set seed for reproducibility
    seed_everything(42)

    try:
        # 1. Prepare Data
        train_sub, val_sub, test_sub = create_demo_subsets(n_samples=20)

        # 2. Configure Environment
        configure_demo_paths(train_sub, val_sub, test_sub)

        # 3. Verify Components
        verify_components()

        # 4. Run Full Pipeline
        run_pipeline_demo()

        print("\n[Demo] All tasks completed successfully.")

    except Exception as e:
        print(f"\n[Demo] FAILED with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
