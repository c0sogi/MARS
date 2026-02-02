import os
import shutil
import torch
import numpy as np
import pandas as pd

# Import from the provided library
from library.config import Config, seed_everything
from library.layers import RobustBlock, DenseDilatedTCN
from library.model import REIDFN
from library.loss import MaskedMCRMSELoss
from library.data import process_data, RNADataset
from library.train import run_training


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup Configuration
    # Redirect outputs to a demo specific directory in working
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    Config.CACHE_DIR = demo_dir
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "demo_submission.csv")
    Config.BATCH_SIZE = 4  # Small batch size for demo
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Verify Layers
    print("\n--- Verifying Layers ---")
    batch_size = 2
    seq_len = 107
    in_channels = 16
    growth_rate = 8

    # Test RobustBlock
    block = RobustBlock(in_channels, growth_rate, dilation=1).to(device)
    dummy_input = torch.randn(batch_size, in_channels, seq_len).to(device)
    output = block(dummy_input)
    # RobustBlock output channel dim is growth_rate (pointwise conv at end)
    assert output.shape == (
        batch_size,
        growth_rate,
        seq_len,
    ), f"RobustBlock output shape mismatch: {output.shape}"
    print("RobustBlock: Verified.")

    # Test DenseDilatedTCN
    dilations = [1, 2, 4]
    tcn = DenseDilatedTCN(in_channels, growth_rate, dilations).to(device)
    tcn_out = tcn(dummy_input)
    # DenseNet output dim = in_channels + num_layers * growth_rate
    expected_dim = in_channels + len(dilations) * growth_rate
    assert tcn_out.shape == (
        batch_size,
        expected_dim,
        seq_len,
    ), f"DenseDilatedTCN output shape mismatch: {tcn_out.shape}"
    print("DenseDilatedTCN: Verified.")

    # 3. Verify Model Architecture (REIDFN)
    print("\n--- Verifying Model (REIDFN) ---")
    model = REIDFN().to(device)

    # Input to REIDFN is (B, 18, L) based on library.data.get_features
    model_input = torch.randn(batch_size, 18, seq_len).to(device)
    # Pair indices: -1 for unpaired. Let's make a dummy pair map.
    pair_indices = torch.full((batch_size, seq_len), -1, dtype=torch.long).to(device)

    # Pass 1: No feedback
    pred1 = model(model_input, pair_indices, prev_preds=None)
    assert pred1.shape == (
        batch_size,
        seq_len,
        5,
    ), f"Model pass 1 output shape mismatch: {pred1.shape}"

    # Pass 2: With feedback
    pred2 = model(model_input, pair_indices, prev_preds=pred1)
    assert pred2.shape == (
        batch_size,
        seq_len,
        5,
    ), f"Model pass 2 output shape mismatch: {pred2.shape}"
    print("REIDFN Model: Verified.")

    # 4. Verify Loss Function
    print("\n--- Verifying Loss (MaskedMCRMSELoss) ---")
    criterion = MaskedMCRMSELoss().to(device)

    # Case A: Identical predictions and targets -> Loss should be 0
    # Create dummy targets matching prediction shape
    dummy_target = pred1.clone().detach()
    loss_zero = criterion(pred1, dummy_target)
    assert torch.isclose(
        loss_zero, torch.tensor(0.0).to(device), atol=1e-6
    ), f"Loss should be 0 for identical inputs, got {loss_zero}"

    # Case B: Known difference
    # Config.SCORED_COLS are indices [0, 1, 3] in the 5-channel output
    # Let's offset index 0 by 1.0.
    # MSE for col 0 = 1.0. MSE for cols 1,3 = 0.0.
    # Mean MSE = (1+0+0)/3 = 0.333...
    # RMSE = sqrt(0.333...) = 0.577...
    dummy_pred_diff = dummy_target.clone()
    dummy_pred_diff[:, : Config.PRED_LEN, 0] += 1.0

    loss_diff = criterion(dummy_pred_diff, dummy_target)
    expected_loss = np.sqrt(1.0 / 3.0)
    assert torch.isclose(
        loss_diff,
        torch.tensor(expected_loss, dtype=torch.float32).to(device),
        rtol=1e-4,
    ), f"Loss calculation mismatch. Expected ~{expected_loss}, got {loss_diff.item()}"
    print("MaskedMCRMSELoss: Verified.")

    # 5. Verify Data Processing
    print("\n--- Verifying Data Processing ---")
    # Use a small debug size to speed up loading
    debug_n = 20
    train_dict, val_dict, test_dict = process_data(
        load_cached_data=False, debug_size=debug_n
    )

    assert len(train_dict["ids"]) == debug_n, "Train data size mismatch"
    assert len(val_dict["ids"]) == debug_n, "Val data size mismatch"
    # Test set might be smaller than debug_n if the file is short, but here check non-empty
    assert len(test_dict["ids"]) > 0, "Test data empty"

    # Verify Dataset class
    ds = RNADataset(train_dict, is_test=False)
    x, pairs, y, _ = ds[0]
    assert x.shape == (18, seq_len), f"Dataset input shape mismatch: {x.shape}"
    assert pairs.shape == (seq_len,), f"Dataset pairs shape mismatch: {pairs.shape}"
    assert y.shape == (seq_len, 5), f"Dataset target shape mismatch: {y.shape}"
    print("Data Processing & Dataset: Verified.")

    # 6. Run Full Training Pipeline (Integration Test)
    print("\n--- Running Full Training Pipeline (1 Epoch, Debug Data) ---")
    # We use the run_training function from library.train
    # This function handles data loading, model init, training loop, validation, and submission

    try:
        run_training(debug_size=32, epochs=1)
    except Exception as e:
        raise RuntimeError(f"Training pipeline failed: {e}")

    # 7. Verify Submission Output
    print("\n--- Verifying Submission Output ---")
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated with shape: {sub_df.shape}")

    # Check columns
    expected_cols = ["id_seqpos"] + Config.ALL_TARGET_COLS
    assert list(sub_df.columns) == expected_cols, "Submission columns mismatch"

    # Check rows count: 32 samples * 107 length = 3424 rows
    # Note: run_training uses debug_size=32 for test set as well
    expected_rows = 32 * Config.SEQ_LEN
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"

    print("Submission Verification: Passed.")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
