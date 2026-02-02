import os
import shutil
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# Import from library
from library.config import Config
from library.utils import set_seed, mcrmse_metric
from library.loss import MCRMSELoss
from library.model import NRDCN
from library.data import process_single_sequence, get_dataloaders
from library.train import train_model


def create_subset_data(source_dir, dest_dir, n_samples=20):
    """Creates small subsets of the metadata CSVs for rapid testing."""
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    files = ["train.csv", "val.csv", "test.csv"]
    new_paths = {}

    for f in files:
        src_path = os.path.join(source_dir, f)
        dest_path = os.path.join(dest_dir, f.replace(".csv", "_subset.csv"))

        # Read only first n_samples + header
        if os.path.exists(src_path):
            df = pd.read_csv(src_path, nrows=n_samples)
            df.to_csv(dest_path, index=False)
            new_paths[f] = dest_path
        else:
            raise FileNotFoundError(f"Source file {src_path} not found.")

    return new_paths


def test_metric_logic():
    """Verifies the MCRMSE metric calculation."""
    print("Verifying MCRMSE metric logic...")
    # Case 1: Perfect prediction
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_pred = np.array([[1.0, 2.0], [3.0, 4.0]])
    score = mcrmse_metric(y_true, y_pred)
    assert score == 0.0, f"Expected 0.0, got {score}"

    # Case 2: Known error
    # Col 0: |1-2|=1, |3-4|=1 -> MSE=1 -> RMSE=1
    # Col 1: |2-2|=0, |4-4|=0 -> MSE=0 -> RMSE=0
    # Mean RMSE = (1 + 0) / 2 = 0.5
    y_pred_off = np.array([[2.0, 2.0], [4.0, 4.0]])
    score = mcrmse_metric(y_true, y_pred_off)
    assert np.isclose(score, 0.5), f"Expected 0.5, got {score}"
    print("Metric logic verified.")


def test_loss_logic():
    """Verifies the MCRMSELoss pytorch module."""
    print("Verifying MCRMSELoss logic...")
    criterion = MCRMSELoss()

    # Config.TARGET_COLS: ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Config.SCORED_COLS: ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    # Scored Indices: 0, 1, 3.

    batch_size = 2
    seq_len = 68
    n_targets = 5

    # Create dummy inputs/targets
    # Predictions are length 107 (full sequence), Targets are length 68 (scored sequence)
    inputs = torch.zeros(batch_size, 107, n_targets)
    targets = torch.zeros(batch_size, seq_len, n_targets)

    # Set targets to 1.0 for scored columns only (indices 0, 1, 3)
    targets[:, :, 0] = 1.0
    targets[:, :, 1] = 1.0
    targets[:, :, 3] = 1.0

    # Inputs are 0.0.
    # Diff is 1.0 for scored columns.
    # MSE per column = 1.0. RMSE per column = 1.0. Mean RMSE = 1.0.

    loss = criterion(inputs, targets)
    assert torch.isclose(
        loss, torch.tensor(1.0)
    ), f"Expected loss 1.0, got {loss.item()}"
    print("Loss logic verified.")


def test_model_forward():
    """Verifies model instantiation and forward pass dimensions."""
    print("Verifying Model forward pass...")
    device = torch.device("cpu")
    model = NRDCN().to(device)
    model.eval()

    B, L = 2, 107
    # Inputs: (B, L, 18) (18 static channels)
    inputs = torch.randn(B, L, 18).to(device)
    # Partner indices: (B, L) with values -1 or 0..L-1
    partner_indices = torch.full((B, L), -1, dtype=torch.long).to(device)
    # Set some dummy pairs
    partner_indices[:, 0] = 5
    partner_indices[:, 5] = 0

    with torch.no_grad():
        # Test cold start (recycling=None)
        out = model(inputs, partner_indices)
        assert out.shape == (B, L, 5), f"Expected shape {(B, L, 5)}, got {out.shape}"

        # Test recycling
        out2 = model(inputs, partner_indices, recycling=out)
        assert out2.shape == (B, L, 5), f"Expected shape {(B, L, 5)}, got {out2.shape}"

    print("Model forward pass verified.")


def run_pipeline_demo():
    """Runs the full training pipeline on a subset of data."""
    print("\nRunning full pipeline demonstration...")

    # 1. Setup paths
    working_dir = "./working/demo_execution"
    metadata_dir = "./metadata"

    # 2. Create subsets (22 samples to allow a batch size of 4 with some batches)
    subset_size = 22
    subset_paths = create_subset_data(metadata_dir, working_dir, n_samples=subset_size)

    # 3. Patch Config for demo execution
    print("Patching Config for demo...")
    Config.WORKING_DIR = working_dir
    Config.TRAIN_CSV = subset_paths["train.csv"]
    Config.VAL_CSV = subset_paths["val.csv"]
    Config.TEST_CSV = subset_paths["test.csv"]

    # Update cache paths to avoid conflict with real run or previous caches
    Config.TRAIN_CACHE = os.path.join(working_dir, "train_data.npz")
    Config.VAL_CACHE = os.path.join(working_dir, "val_data.npz")
    Config.TEST_CACHE = os.path.join(working_dir, "test_data.npz")

    # Clear stale cache to ensure subset data is processed
    for cache_path in [Config.TRAIN_CACHE, Config.VAL_CACHE, Config.TEST_CACHE]:
        if os.path.exists(cache_path):
            os.remove(cache_path)

    Config.MODEL_PATH = os.path.join(working_dir, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(working_dir, "submission.csv")

    # Optimize hyperparameters for speed
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.PATIENCE = 1

    # 4. Run Training
    # This calls get_dataloaders -> preprocess_data -> saves cache -> trains -> infers -> saves submission
    train_model()

    # 5. Verify Output
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not generated.")

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated with shape: {df_sub.shape}")

    # Verify expected rows: subset_size samples * 107 seq len
    expected_rows = subset_size * 107
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(df_sub)}"

    print("Pipeline demonstration completed successfully.")


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(42)

    # Run Unit Tests
    test_metric_logic()
    test_loss_logic()
    test_model_forward()

    # Run Integration/Pipeline Test
    run_pipeline_demo()
