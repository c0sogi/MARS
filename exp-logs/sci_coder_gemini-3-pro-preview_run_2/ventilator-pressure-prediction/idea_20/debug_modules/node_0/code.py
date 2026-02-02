import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import library components
from library.config import Config
from library.utils import seed_everything, compute_mae
from library.dataset import load_data, VentilatorDataset
from library.model import CWCDP_BiLSTM
from library.loss import WeightedL1Loss
from library.train import train_model
from library.inference import generate_predictions


def run_demonstration():
    print("============================================================")
    print("       Ventilator Pressure Prediction Library Demo          ")
    print("============================================================")

    # ------------------------------------------------------------------
    # 1. Configuration Override for Speed & Debugging
    # ------------------------------------------------------------------
    print("\n[Step 1] Configuring environment for rapid demonstration...")

    # Modify Config attributes to run a fast debug session
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 100  # Use only 100 breaths
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 16  # Smaller batch size for small dataset
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Ensure working directory is clean for this run (optional but good for demo)
    if os.path.exists(Config.WORKING_DIR):
        # We don't delete the whole dir to avoid removing pre-existing large files if any,
        # but we ensure we are ready to write new ones.
        pass

    seed_everything(Config.SEED)
    print("Configuration updated: DEBUG=True, EPOCHS=1, SAMPLE_SIZE=100")

    # ------------------------------------------------------------------
    # 2. Verify Dataset Loading & Processing
    # ------------------------------------------------------------------
    print("\n[Step 2] Verifying Data Loading and Processing...")

    # Force reload to ensure we process the small debug subset
    # We remove the debug cache if it exists to verify the processing pipeline
    debug_cache_path = Config.CACHE_TRAIN_PATH.replace(".parquet", "_debug.npz")
    if os.path.exists(debug_cache_path):
        os.remove(debug_cache_path)

    train_dataset = load_data("train", debug=True, load_cached_data=False)

    # Assertions
    assert isinstance(
        train_dataset, VentilatorDataset
    ), "Failed to return VentilatorDataset instance"
    assert (
        len(train_dataset) == Config.DEBUG_SAMPLE_SIZE
    ), f"Dataset size mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(train_dataset)}"

    sample = train_dataset[0]
    # Shape checks: (Seq_Len, Features)
    seq_len = 80
    assert sample["x"].shape == (
        seq_len,
        Config.INPUT_DIM,
    ), f"Input shape mismatch. Expected ({seq_len}, {Config.INPUT_DIM}), got {sample['x'].shape}"
    assert sample["y"].shape == (
        seq_len,
    ), f"Target shape mismatch. Expected ({seq_len},), got {sample['y'].shape}"
    assert sample["u_out"].shape == (
        seq_len,
    ), f"Control shape mismatch. Expected ({seq_len},), got {sample['u_out'].shape}"

    print("Dataset loaded and verified successfully.")

    # ------------------------------------------------------------------
    # 3. Verify Model Architecture
    # ------------------------------------------------------------------
    print("\n[Step 3] Verifying Model Architecture (CWCDP-BiLSTM)...")

    model = CWCDP_BiLSTM()
    model.to(Config.DEVICE)
    model.eval()

    # Create dummy input batch: (Batch_Size, Seq_Len, Input_Dim)
    dummy_input = torch.randn(2, 80, Config.INPUT_DIM).to(Config.DEVICE)

    with torch.no_grad():
        output = model(dummy_input)

    # Expected output: (Batch_Size, Seq_Len)
    assert output.shape == (
        2,
        80,
    ), f"Model output shape mismatch. Expected (2, 80), got {output.shape}"

    print("Model instantiated and forward pass verified.")

    # ------------------------------------------------------------------
    # 4. Verify Custom Loss Function Logic
    # ------------------------------------------------------------------
    print("\n[Step 4] Verifying WeightedL1Loss Logic...")

    # Weights: Insp=1.0, Exp=0.1
    criterion = WeightedL1Loss(inspiratory_weight=1.0, expiratory_weight=0.1)

    # Case A: Pure Inspiratory (u_out = 0)
    preds_a = torch.tensor([10.0, 10.0])
    targets_a = torch.tensor([12.0, 12.0])  # Diff = 2.0
    u_out_a = torch.tensor([0.0, 0.0])

    loss_a = criterion(preds_a, targets_a, u_out_a)
    expected_loss_a = 2.0 * 1.0
    assert torch.isclose(
        loss_a, torch.tensor(expected_loss_a)
    ), f"Inspiratory loss incorrect. Expected {expected_loss_a}, got {loss_a.item()}"

    # Case B: Pure Expiratory (u_out = 1)
    preds_b = torch.tensor([10.0, 10.0])
    targets_b = torch.tensor([12.0, 12.0])  # Diff = 2.0
    u_out_b = torch.tensor([1.0, 1.0])

    loss_b = criterion(preds_b, targets_b, u_out_b)
    expected_loss_b = 2.0 * 0.1
    assert torch.isclose(
        loss_b, torch.tensor(expected_loss_b)
    ), f"Expiratory loss incorrect. Expected {expected_loss_b}, got {loss_b.item()}"

    print("WeightedL1Loss logic verified.")

    # ------------------------------------------------------------------
    # 5. Verify Metric Calculation (MAE)
    # ------------------------------------------------------------------
    print("\n[Step 5] Verifying Compute MAE (Inspiratory Phase Only)...")

    # 4 samples: 2 insp, 2 exp
    # Insp errors: |10-12|=2, |20-21|=1 -> Mean = 1.5
    # Exp errors: |5-10|=5, |5-10|=5 -> Should be ignored
    y_pred = torch.tensor([10.0, 20.0, 5.0, 5.0])
    y_true = torch.tensor([12.0, 21.0, 10.0, 10.0])
    u_out = torch.tensor([0.0, 0.0, 1.0, 1.0])

    mae = compute_mae(y_pred, y_true, u_out)
    expected_mae = 1.5

    assert np.isclose(
        mae, expected_mae
    ), f"MAE calculation incorrect. Expected {expected_mae}, got {mae}"
    print("Metric calculation verified.")

    # ------------------------------------------------------------------
    # 6. Execute Training Loop
    # ------------------------------------------------------------------
    print("\n[Step 6] Executing Training Loop (1 Epoch)...")

    # This function handles loading data, initializing model/optimizer, and loop
    train_model(debug=True, load_cached_data=True)

    # Verify artifact creation
    assert os.path.exists(
        Config.MODEL_PATH
    ), "Model file (best_model.pth) was not created."
    print("Training loop completed and model saved.")

    # ------------------------------------------------------------------
    # 7. Execute Inference Pipeline
    # ------------------------------------------------------------------
    print("\n[Step 7] Executing Inference Pipeline...")

    # This generates predictions on the test set (debug subset)
    generate_predictions(debug=True, load_cached_data=True)

    # Verify submission file
    submission_path = Config.SUBMISSION_PATH.replace(".csv", "_debug.csv")
    assert os.path.exists(
        submission_path
    ), f"Submission file not found at {submission_path}"

    df_sub = pd.read_csv(submission_path)

    # Expected rows: DEBUG_SAMPLE_SIZE * 80 time steps
    expected_rows = Config.DEBUG_SAMPLE_SIZE * 80
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    assert (
        "id" in df_sub.columns and "pressure" in df_sub.columns
    ), "Submission columns missing."

    print(f"Inference completed. Submission generated with {len(df_sub)} rows.")
    print("============================================================")
    print("       Demonstration Completed Successfully                 ")
    print("============================================================")


if __name__ == "__main__":
    run_demonstration()
