import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, compute_mcrmse
from library.loss import MaskedMSELoss
from library.data import get_dataloaders
from library.model import RNAModel
from library.train import Trainer


def main():
    print("Starting Demonstration Script...")

    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    print("\n[1] Configuring environment for demo run...")

    # Modify Config for a fast demonstration
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Create directories
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Reduce compute load for demo
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 8  # Smaller batch size for the small debug dataset

    # Set reproducible seed
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # --------------------------------------------------------------------------
    # 2. Data Loading & Verification
    # --------------------------------------------------------------------------
    print("\n[2] Verifying Data Loading...")

    # Use debug_size to load only 50 samples for speed
    debug_size = 50
    # Force reload to ensure we use the debug subset and not a cached full dataset
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, debug_size=debug_size
    )

    print(f"    Train batches: {len(train_loader)}")
    print(f"    Val batches: {len(val_loader)}")
    print(f"    Test batches: {len(test_loader)}")

    # Verify Train Batch Structure
    batch = next(iter(train_loader))
    required_keys = ["seq", "loop", "dist", "targets", "id"]
    for key in required_keys:
        assert key in batch, f"Missing key {key} in batch"

    # Check dimensions
    # seq: (B, 107), targets: (B, 107, 3)
    seq = batch["seq"].to(device)
    loop = batch["loop"].to(device)
    dist = batch["dist"].to(device)
    targets = batch["targets"].to(device)

    assert seq.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), f"Seq shape mismatch: {seq.shape}"
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        3,
    ), f"Targets shape mismatch: {targets.shape}"

    print("    Batch structure verified successfully.")

    # --------------------------------------------------------------------------
    # 3. Model Instantiation & Forward Pass
    # --------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    model = RNAModel().to(device)

    # Run forward pass
    preds = model(seq, loop, dist)

    # Check output shape: (Batch, Seq_Len, 3)
    assert preds.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        3,
    ), f"Prediction shape mismatch: {preds.shape}"

    print("    Forward pass successful. Output shape matches expected dimensions.")

    # --------------------------------------------------------------------------
    # 4. Loss Function Verification
    # --------------------------------------------------------------------------
    print("\n[4] Verifying Loss Function...")

    criterion = MaskedMSELoss()
    loss = criterion(preds, targets)

    assert isinstance(loss, torch.Tensor), "Loss must be a tensor"
    assert loss.dim() == 0, "Loss must be a scalar"
    assert loss.item() >= 0, "Loss must be non-negative"

    print(f"    Loss calculation successful: {loss.item():.6f}")

    # --------------------------------------------------------------------------
    # 5. Metric Logic Verification
    # --------------------------------------------------------------------------
    print("\n[5] Verifying Metric Calculation (MCRMSE)...")

    # Create dummy data: 2 samples, sequence length 5 (scored), 2 columns
    # Note: compute_mcrmse expects (N, Seq, Cols) or flattened.
    # The utils implementation handles arbitrary shapes as long as they match.

    # Let's simulate the exact shape used in validation: (N, 68, 3)
    dummy_preds = np.zeros((10, 68, 3))
    dummy_targets = np.zeros((10, 68, 3))

    # Set column 0 error to 1.0 (MSE=1.0, RMSE=1.0)
    dummy_preds[:, :, 0] = dummy_targets[:, :, 0] + 1.0
    # Set column 1 error to 2.0 (MSE=4.0, RMSE=2.0)
    dummy_preds[:, :, 1] = dummy_targets[:, :, 1] + 2.0
    # Set column 2 error to 0.0 (MSE=0.0, RMSE=0.0)

    expected_mcrmse = (1.0 + 2.0 + 0.0) / 3.0  # Average of RMSEs
    calculated_mcrmse = compute_mcrmse(dummy_preds, dummy_targets)

    assert np.isclose(
        calculated_mcrmse, expected_mcrmse
    ), f"MCRMSE mismatch: expected {expected_mcrmse}, got {calculated_mcrmse}"

    print(f"    MCRMSE verification passed. Value: {calculated_mcrmse}")

    # --------------------------------------------------------------------------
    # 6. Training Loop Demonstration
    # --------------------------------------------------------------------------
    print("\n[6] Running Training Loop (Demo)...")

    trainer = Trainer(model, device, train_loader, val_loader, test_loader)
    trainer.fit()

    assert os.path.exists(Config.MODEL_PATH), "Best model checkpoint was not saved."
    print("    Training complete and model saved.")

    # --------------------------------------------------------------------------
    # 7. Submission Generation
    # --------------------------------------------------------------------------
    print("\n[7] Generating Submission...")

    trainer.generate_submission()

    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file not found."

    # Verify Submission Content
    df_sub = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"    Submission shape: {df_sub.shape}")

    # Expected rows: Number of test samples * 107 positions
    # We used debug_size=50 for loading, so test_loader has 50 samples.
    expected_rows = debug_size * Config.SEQ_LEN
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch: expected {expected_rows}, got {len(df_sub)}"

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
    ), f"Submission columns mismatch: {list(df_sub.columns)}"

    # Check values for unscored columns (should be 0.0 as per logic)
    assert (df_sub["deg_pH10"] == 0.0).all(), "deg_pH10 should be 0.0"
    assert (df_sub["deg_50C"] == 0.0).all(), "deg_50C should be 0.0"

    print("    Submission verification passed.")

    print("\nAll demonstration steps completed successfully.")


if __name__ == "__main__":
    main()
