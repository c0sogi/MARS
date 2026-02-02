import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config
from library.features import prepare_datasets
from library.dataset import get_data_loaders
from library.model import TransLSTM
from library.utils import MaskedL1Loss, seed_everything
from library.trainer import Trainer


def main():
    # --- 1. Setup & Configuration Override ---
    print(">>> Setting up configuration for demonstration...")

    # Override Config for a fast demo run
    Config.DEBUG = True  # Use small subset of data
    Config.EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 16  # Small batch size for debug data
    Config.EXP_NAME = "demo_run"  # Separate output directory
    Config.OUTPUT_DIR = f"./working/{Config.EXP_NAME}/"

    # Re-run setup to ensure output directory exists
    Config.setup()

    # Set seed for reproducibility
    seed_everything(Config.SEED)

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Output Directory: {Config.OUTPUT_DIR}")

    # --- 2. Data Preparation & Verification ---
    print("\n>>> Preparing datasets (Feature Engineering)...")

    # Force processing from scratch (load_cached_data=False) to ensure DEBUG slicing is applied
    train_data, val_data, test_data = prepare_datasets(load_cached_data=False)

    # Verify Data Dictionary Structure
    assert (
        "x" in train_data and "y" in train_data and "u_out" in train_data
    ), "Train data missing keys"
    assert (
        "x" in test_data and "ids" in test_data and "u_out" in test_data
    ), "Test data missing keys"

    # Verify Shapes (Seq_Len should be 80)
    # x shape: (N_breaths, 80, N_features)
    print(f"Train X Shape: {train_data['x'].shape}")
    assert train_data["x"].ndim == 3
    assert train_data["x"].shape[1] == Config.SEQ_LEN
    assert train_data["y"].shape == (train_data["x"].shape[0], Config.SEQ_LEN)

    # Verify DataLoaders
    print("\n>>> Verifying DataLoaders...")
    train_loader, val_loader, test_loader = get_data_loaders(load_cached_data=True)

    # Fetch one batch to verify tensor shapes
    x_batch, u_out_batch, y_batch = next(iter(train_loader))
    print(f"Batch X Shape: {x_batch.shape}")
    print(f"Batch Y Shape: {y_batch.shape}")

    assert x_batch.shape[0] == Config.BATCH_SIZE
    assert x_batch.shape[1] == Config.SEQ_LEN
    assert y_batch.shape == (Config.BATCH_SIZE, Config.SEQ_LEN)

    # --- 3. Model Logic Verification ---
    print("\n>>> Verifying Model Architecture...")
    model = TransLSTM()
    model.eval()

    # Create a dummy input matching the batch shape
    # Input features = Cont features + R_cat + C_cat
    n_features = train_data["x"].shape[2]
    dummy_input = torch.randint(
        0, 3, (Config.BATCH_SIZE, Config.SEQ_LEN, n_features)
    ).float()

    # The last two columns are categorical indices (0, 1, 2)
    # Ensure they are within range for Embedding layers
    dummy_input[:, :, -2] = torch.randint(
        0, 3, (Config.BATCH_SIZE, Config.SEQ_LEN)
    ).float()
    dummy_input[:, :, -1] = torch.randint(
        0, 3, (Config.BATCH_SIZE, Config.SEQ_LEN)
    ).float()

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    # Expected output: [Batch, Seq_Len]
    assert output.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), f"Expected output shape {(Config.BATCH_SIZE, Config.SEQ_LEN)}, got {output.shape}"

    # --- 4. Loss Function Verification ---
    print("\n>>> Verifying Masked L1 Loss...")
    criterion = MaskedL1Loss()

    # Case 1: Perfect prediction during inspiration (u_out=0), error during expiration (u_out=1)
    # Loss should be 0 because expiration is masked out.
    pred = torch.tensor([[10.0, 10.0], [10.0, 10.0]])
    target = torch.tensor([[10.0, 20.0], [10.0, 20.0]])
    u_out = torch.tensor([[0, 1], [0, 1]])  # 1st step inspiration, 2nd step expiration

    loss = criterion(pred, target, u_out)
    print(f"Loss (Perfect Insp, Bad Exp): {loss.item()}")
    assert torch.isclose(
        loss, torch.tensor(0.0)
    ), "Loss should be 0 when error is only in masked region"

    # Case 2: Error during inspiration
    target_err = torch.tensor([[12.0, 20.0], [12.0, 20.0]])  # Error of 2.0 at index 0
    loss_err = criterion(pred, target_err, u_out)
    # MAE on valid steps: (|10-12| + |10-12|) / 2 = 2.0
    print(f"Loss (Error Insp): {loss_err.item()}")
    assert torch.isclose(
        loss_err, torch.tensor(2.0)
    ), "Loss calculation incorrect for inspiratory phase"

    # --- 5. Training & Inference Loop ---
    print("\n>>> Starting Training Loop (Demo)...")
    trainer = Trainer()

    # Run fit (uses the loaders we prepared implicitly via cache or reload)
    # We pass load_cached_data=True because we generated the cache in step 2
    trainer.fit(load_cached_data=True)

    # --- 6. Output Verification ---
    print("\n>>> Verifying Output Files...")

    model_path = os.path.join(Config.OUTPUT_DIR, "model.pth")
    submission_path = os.path.join(Config.OUTPUT_DIR, "submission.csv")
    final_sub_path = "./submission/submission.csv"

    if os.path.exists(model_path):
        print(f"Verified: Model saved at {model_path}")
    else:
        raise FileNotFoundError(f"Model not found at {model_path}")

    if os.path.exists(submission_path):
        print(f"Verified: Submission saved at {submission_path}")

        # Check submission content format
        df_sub = pd.read_csv(submission_path)
        print(f"Submission Shape: {df_sub.shape}")
        assert list(df_sub.columns) == ["id", "pressure"], "Submission columns mismatch"
        assert not df_sub.isnull().values.any(), "Submission contains NaNs"
        # Test set in debug mode: 80 timesteps * 50 breaths = 4000 rows
        # (Exact number depends on how features.py slices, usually 80*50)
        expected_rows = 80 * 50
        assert (
            len(df_sub) == expected_rows
        ), f"Expected {expected_rows} rows, got {len(df_sub)}"
    else:
        raise FileNotFoundError(f"Submission not found at {submission_path}")

    if os.path.exists(final_sub_path):
        print(f"Verified: Final submission copy at {final_sub_path}")
    else:
        raise FileNotFoundError(f"Final submission not found at {final_sub_path}")

    print("\n>>> Demonstration Completed Successfully.")


if __name__ == "__main__":
    main()
