import os
import sys
import shutil
import torch
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Import library components
from library.config import Config
from library.utils import seed_everything, mcrmse_loss
from library.data import get_dataloaders, get_bond_type_tokens, get_distance_vector
from library.model import BondAwareModel
from library.train import run_training, generate_submission


def run_demo():
    print(">>> Starting RNA Degradation Pipeline Demo")

    # ------------------------------------------------------------------------
    # 1. Configuration Override
    # ------------------------------------------------------------------------
    print("\n[1] Configuring environment for demo run...")

    # Set paths to a separate demo directory to avoid messing with real experiments
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config parameters for speed and isolation
    Config.working_dir = demo_dir
    Config.debug = True  # Use small subset of data
    Config.epochs = 1
    Config.batch_size = 4
    Config.num_workers = 0  # Avoid multiprocessing overhead for small demo

    # Update cache paths to point to demo directory
    Config.train_cache = os.path.join(demo_dir, "train_data_debug.pt")
    Config.val_cache = os.path.join(demo_dir, "val_data_debug.pt")
    Config.test_cache = os.path.join(demo_dir, "test_data.pt")

    # Update model and submission paths
    Config.model_path = os.path.join(demo_dir, "best_model.pth")
    Config.submission_path = os.path.join(demo_dir, "submission.csv")

    # Set seed
    seed_everything(Config.seed)
    print(f"    Working directory set to: {Config.working_dir}")
    print(f"    Debug mode: {Config.debug}")

    # ------------------------------------------------------------------------
    # 2. Logic Verification: Utilities
    # ------------------------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")

    # Test MCRMSE Loss
    # Create dummy targets and preds.
    # Target: [[0, 0, 0], [1, 1, 1]]
    # Pred:   [[1, 1, 1], [2, 2, 2]]
    # Diff is 1 everywhere. Squared diff is 1. Mean is 1. RMSE is 1.
    y_true = torch.tensor(
        [[[0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0]]], dtype=torch.float32
    )  # (2, 1, 3)
    y_pred = torch.tensor(
        [[[1.0, 1.0, 1.0]], [[2.0, 2.0, 2.0]]], dtype=torch.float32
    )  # (2, 1, 3)
    loss = mcrmse_loss(y_true, y_pred)

    assert torch.isclose(
        loss, torch.tensor(1.0)
    ), f"MCRMSE calculation incorrect. Expected 1.0, got {loss.item()}"
    print("    mcrmse_loss: Passed")

    # ------------------------------------------------------------------------
    # 3. Logic Verification: Data Processing
    # ------------------------------------------------------------------------
    print("\n[3] Verifying Data Processing Logic...")

    # Test Sequence: "AGCU"
    # Test Structure: "(..)" -> A pairs with U (indices 0 and 3)
    # G (1) and C (2) are unpaired
    seq = "AGCU"
    struct = "(..)"

    # Expected Bond Tokens:
    # 0 (A): Paired with U. "A-U" -> Config.bond2id["A-U"] -> 0
    # 1 (G): Unpaired -> 7
    # 2 (C): Unpaired -> 7
    # 3 (U): Paired with A. "U-A" -> Config.bond2id["U-A"] -> 1
    expected_bonds = [0, 7, 7, 1]

    bonds = get_bond_type_tokens(seq, struct)
    np.testing.assert_array_equal(bonds, np.array(expected_bonds))
    print("    get_bond_type_tokens: Passed")

    # Expected Distance Vector:
    # 0: paired with 3 -> dist = 3 - 0 = 3.0
    # 1: unpaired -> 0.0
    # 2: unpaired -> 0.0
    # 3: paired with 0 -> dist = 0 - 3 = -3.0
    expected_dist = [3.0, 0.0, 0.0, -3.0]
    dists = get_distance_vector(struct)
    np.testing.assert_array_equal(dists, np.array(expected_dist))
    print("    get_distance_vector: Passed")

    # ------------------------------------------------------------------------
    # 4. Pipeline Integration: Data Loaders
    # ------------------------------------------------------------------------
    print("\n[4] Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(debug=Config.debug)

    print(f"    Train Batches: {len(train_loader)}")
    print(f"    Val Batches:   {len(val_loader)}")

    # Verify Batch Structure
    batch = next(iter(train_loader))
    required_keys = ["seq", "loop", "bond", "dist", "targets", "id"]
    for key in required_keys:
        assert key in batch, f"Batch missing key: {key}"

    # Verify Shapes
    # Seq: (B, 107)
    assert batch["seq"].shape == (
        Config.batch_size,
        Config.seq_len,
    ), f"Incorrect sequence shape: {batch['seq'].shape}"
    # Targets: (B, 68, 3) - Note: Data loader provides full processed targets,
    # but based on data.py, it stacks Config.target_cols.
    # Config.target_cols has 3 items. Length is 68.
    assert batch["targets"].shape == (
        Config.batch_size,
        Config.pred_len,
        Config.num_targets,
    ), f"Incorrect target shape: {batch['targets'].shape}"

    print("    DataLoader shapes verified.")

    # ------------------------------------------------------------------------
    # 5. Model Verification
    # ------------------------------------------------------------------------
    print("\n[5] Verifying Model Architecture...")
    device = torch.device(Config.device)
    model = BondAwareModel().to(device)

    # Move batch to device
    seq = batch["seq"].to(device)
    loop = batch["loop"].to(device)
    bond = batch["bond"].to(device)
    dist = batch["dist"].to(device)

    # Forward Pass
    output = model(seq, loop, bond, dist)

    # Check Output Shape: (B, 107, 3)
    # The model outputs predictions for the full sequence length (107),
    # slicing happens in the loss function.
    expected_shape = (Config.batch_size, Config.seq_len, 3)
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"

    print(f"    Forward pass successful. Output shape: {output.shape}")

    # ------------------------------------------------------------------------
    # 6. Training Simulation
    # ------------------------------------------------------------------------
    print("\n[6] Running Training Simulation (1 Epoch)...")
    # This function uses the Config we modified globally
    run_training()

    # Verify checkpoint creation
    assert os.path.exists(Config.model_path), "Model checkpoint was not created."
    print("    Training complete. Checkpoint verified.")

    # ------------------------------------------------------------------------
    # 7. Submission Generation
    # ------------------------------------------------------------------------
    print("\n[7] Generating Submission...")
    generate_submission()

    # Verify submission file
    assert os.path.exists(Config.submission_path), "Submission file was not created."

    sub_df = pd.read_csv(Config.submission_path)
    print(f"    Submission loaded. Shape: {sub_df.shape}")

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
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Got {list(sub_df.columns)}"

    # Verify row count
    # In debug mode, get_dataloaders slices the dataset.
    # subset_size = Config.batch_size * 2 = 8 samples.
    # Each sample has 107 positions.
    # Total rows = 8 * 107 = 856.
    expected_rows = (Config.batch_size * 2) * Config.seq_len
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"

    print("    Submission verification passed.")
    print("\n>>> Demo Completed Successfully.")


if __name__ == "__main__":
    run_demo()
