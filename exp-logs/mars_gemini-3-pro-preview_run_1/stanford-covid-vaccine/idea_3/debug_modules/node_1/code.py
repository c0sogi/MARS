import os
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, mcrmse_metric
from library.dataset import RNADataset
from library.model import RNATransformer, masked_mse_loss
from library.engine import Engine


def main():
    # 1. Setup
    warnings.filterwarnings("ignore")
    seed_everything(Config.SEED)
    print("=== Starting Demonstration Script ===")

    # Ensure working directory exists (Config handles this, but good to be sure for checks)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Dataset Verification
    print("\n--- Verifying RNADataset ---")
    # Initialize dataset (this will process parquet files since cache might not exist yet)
    train_dataset = RNADataset(split="train", load_cached_data=False)

    print(f"Train dataset size: {len(train_dataset)}")
    assert len(train_dataset) > 0, "Train dataset should not be empty."

    # Fetch one sample
    sample = train_dataset[0]

    # Check keys
    expected_keys = {"id", "sequence", "structure", "loop", "targets", "mask"}
    assert expected_keys.issubset(
        sample.keys()
    ), f"Missing keys in sample. Found: {sample.keys()}"

    # Check shapes
    seq_len = Config.SEQ_LEN  # 107
    num_targets = Config.NUM_TARGETS  # 5

    assert sample["sequence"].shape == (
        seq_len,
    ), f"Sequence shape mismatch: {sample['sequence'].shape}"
    assert sample["targets"].shape == (
        seq_len,
        num_targets,
    ), f"Targets shape mismatch: {sample['targets'].shape}"
    assert sample["mask"].shape == (
        seq_len,
    ), f"Mask shape mismatch: {sample['mask'].shape}"

    # Check Mask Logic: First 68 should be 1.0, rest 0.0
    pred_len = Config.PRED_LEN  # 68
    mask_active = sample["mask"][:pred_len]
    mask_inactive = sample["mask"][pred_len:]

    assert torch.all(mask_active == 1.0), "First 68 positions in mask must be 1.0"
    assert torch.all(mask_inactive == 0.0), "Positions > 68 in mask must be 0.0"
    print("Dataset shapes and masking logic verified.")

    # 3. Model Verification
    print("\n--- Verifying RNATransformer Model ---")
    model = RNATransformer()
    model.eval()

    # Create dummy batch
    batch_size = 4
    dummy_seq = torch.randint(0, Config.VOCAB_SIZE_SEQ, (batch_size, seq_len))
    dummy_struct = torch.randint(0, Config.VOCAB_SIZE_STRUCT, (batch_size, seq_len))
    dummy_loop = torch.randint(0, Config.VOCAB_SIZE_LOOP, (batch_size, seq_len))

    with torch.no_grad():
        output = model(dummy_seq, dummy_struct, dummy_loop)

    print(f"Model output shape: {output.shape}")
    assert output.shape == (
        batch_size,
        seq_len,
        num_targets,
    ), f"Expected output shape {(batch_size, seq_len, num_targets)}, got {output.shape}"
    print("Model forward pass verified.")

    # 4. Loss Function Verification
    print("\n--- Verifying masked_mse_loss ---")
    # Case: Preds = 1, Targets = 0, Mask = 1. MSE should be 1.0
    preds_t = torch.ones((2, seq_len, num_targets), dtype=torch.float32)
    targets_t = torch.zeros((2, seq_len, num_targets), dtype=torch.float32)
    mask_t = torch.ones((2, seq_len), dtype=torch.float32)

    loss = masked_mse_loss(preds_t, targets_t, mask_t)
    print(f"Calculated Loss (All ones vs Zeros): {loss.item()}")
    assert np.isclose(loss.item(), 1.0, atol=1e-5), "Loss should be 1.0"

    # Case: Mask half the sequence. Loss should still be 1.0 (average over active elements)
    mask_t[:, 50:] = 0.0
    loss_masked = masked_mse_loss(preds_t, targets_t, mask_t)
    print(f"Calculated Loss (Half masked): {loss_masked.item()}")
    assert np.isclose(
        loss_masked.item(), 1.0, atol=1e-5
    ), "Masked loss should still be 1.0"
    print("Loss function verified.")

    # 5. Metric Verification
    print("\n--- Verifying mcrmse_metric ---")
    # MCRMSE only scores columns 0, 1, 3 and first 68 positions.
    # Let's make error = 1.0 for scored columns/positions, and error = 100.0 for others.
    # The result should be exactly 1.0 if filtering works.

    y_true = torch.zeros((1, seq_len, num_targets))
    y_pred = torch.zeros((1, seq_len, num_targets))

    # Add error to scored columns (0, 1, 3) at valid positions (0-67)
    # Scored cols: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_indices = [0, 1, 3]
    y_pred[:, :68, scored_indices] = 1.0

    # Add huge error to unscored columns/positions
    y_pred[:, 68:, :] = 100.0  # Unscored positions
    y_pred[:, :68, 2] = 100.0  # Unscored column 2
    y_pred[:, :68, 4] = 100.0  # Unscored column 4

    metric_val = mcrmse_metric(y_true, y_pred)
    print(f"Metric Value: {metric_val}")
    assert np.isclose(
        metric_val, 1.0, atol=1e-5
    ), f"Metric should be 1.0, ignoring unscored errors. Got {metric_val}"
    print("Metric logic verified.")

    # 6. Engine Execution (Training & Inference)
    print("\n--- Running Engine (Debug Mode) ---")
    # Initialize Engine
    engine = Engine()

    # Run training for 1 epoch on a subset (debug=True)
    # This tests the training loop, validation loop, and checkpoint saving
    engine.run_training(epochs=1, debug=True)

    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint was not saved."
    print("Training loop completed and model saved.")

    # Run Inference
    print("Running Inference...")
    engine.inference()

    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not created."
    print("Inference completed.")

    # 7. Verify Submission File
    print("\n--- Verifying Submission File ---")
    df_sub = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Submission shape: {df_sub.shape}")
    print(f"Submission columns: {df_sub.columns.tolist()}")

    # Check columns
    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    assert list(df_sub.columns) == expected_cols, "Submission columns mismatch."

    # Check row count
    # Test set has 240 samples. Each has 107 positions. Total rows = 240 * 107 = 25680
    n_test_samples = 240
    expected_rows = n_test_samples * seq_len
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(df_sub)}"

    # Check ID format (id_xxxxx_0)
    assert df_sub["id_seqpos"].iloc[0].endswith("_0"), "First row ID should end with _0"
    assert (
        df_sub["id_seqpos"].iloc[106].endswith("_106")
    ), "107th row ID should end with _106"

    print("Submission file format verified.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
