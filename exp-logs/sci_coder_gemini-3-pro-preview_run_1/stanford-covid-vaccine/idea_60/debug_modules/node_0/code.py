import os
import shutil
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.dataset import get_dataset, RNADataset
from library.model import RNAModel
from library.loss import MaskedMSELoss
from library.utils import seed_everything, calculate_mcrmse
from library.train import run_training


def run_demo():
    print("=== Starting RNA Degradation Library Demo ===")

    # 1. Setup Environment
    # ----------------------------------------------------------------
    seed_everything(42)

    # Define a separate cache directory for this demo to avoid conflicts
    demo_cache_dir = "./working/demo_execution"
    if os.path.exists(demo_cache_dir):
        shutil.rmtree(demo_cache_dir)
    os.makedirs(demo_cache_dir, exist_ok=True)

    # Override Config for the demo
    Config.CACHE_DIR = demo_cache_dir
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    print(f"Working directory: {demo_cache_dir}")

    # 2. Data Loading & Processing Verification
    # ----------------------------------------------------------------
    print("\n[1/5] Verifying Data Processing...")

    # Load training data (using the provided metadata parquet)
    # We force load_cached_data=False to demonstrate processing logic
    ids, X_seq, X_loop, X_dist, y = get_dataset(
        Config.TRAIN_PATH, mode="train", load_cached_data=False
    )

    # Verify Shapes
    # Sequence length should be 107
    # Targets should be (N, 68, 3) corresponding to reactivity, deg_Mg_pH10, deg_Mg_50C
    print(f"  - X_seq shape: {X_seq.shape}")
    print(f"  - y shape: {y.shape}")

    assert (
        X_seq.shape[1] == Config.SEQ_LEN
    ), f"Expected sequence length {Config.SEQ_LEN}, got {X_seq.shape[1]}"
    assert (
        y.shape[1] == Config.PRED_LEN
    ), f"Expected target length {Config.PRED_LEN}, got {y.shape[1]}"
    assert y.shape[2] == 3, f"Expected 3 target channels, got {y.shape[2]}"

    # Create a small dataset/loader for model testing
    dataset = RNADataset(X_seq[:10], X_loop[:10], X_dist[:10], y[:10])
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    batch_seq, batch_loop, batch_dist, batch_y = next(iter(loader))
    print("  - DataLoader batch retrieval successful.")

    # 3. Model Architecture Verification
    # ----------------------------------------------------------------
    print("\n[2/5] Verifying Model Architecture...")

    device = "cpu"  # Use CPU for simple logic verification
    model = RNAModel(Config).to(device)
    model.eval()

    with torch.no_grad():
        # Forward pass
        preds = model(
            batch_seq.to(device), batch_loop.to(device), batch_dist.to(device)
        )

    print(f"  - Prediction shape: {preds.shape}")

    # Model outputs predictions for the full sequence length (107)
    assert preds.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        3,
    ), f"Expected output shape {(Config.BATCH_SIZE, Config.SEQ_LEN, 3)}, got {preds.shape}"

    print("  - Forward pass successful.")

    # 4. Loss Function Verification
    # ----------------------------------------------------------------
    print("\n[3/5] Verifying MaskedMSELoss...")

    criterion = MaskedMSELoss(scored_len=Config.PRED_LEN)

    # Create dummy predictions (B, 107, 3) and targets (B, 68, 3)
    # We make predictions exactly 1.0 greater than targets to make math easy
    dummy_targets = torch.zeros((2, 68, 3))
    dummy_preds = torch.zeros((2, 107, 3))
    dummy_preds[:, :68, :] = 1.0  # The scored part has error 1.0
    dummy_preds[:, 68:, :] = (
        100.0  # The unscored part has huge error (should be ignored)
    )

    loss = criterion(dummy_preds, dummy_targets)

    # MSE of 1.0 is 1.0.
    expected_loss = 1.0
    print(f"  - Calculated Loss: {loss.item()}")

    assert np.isclose(
        loss.item(), expected_loss, atol=1e-6
    ), f"Expected loss {expected_loss}, got {loss.item()}"

    print("  - MaskedMSELoss logic verified (unscored tail ignored).")

    # 5. Metric Verification (MCRMSE)
    # ----------------------------------------------------------------
    print("\n[4/5] Verifying MCRMSE Metric...")

    # Scenario:
    # Col 0: Error 1.0 -> MSE=1.0 -> RMSE=1.0
    # Col 1: Error 2.0 -> MSE=4.0 -> RMSE=2.0
    # Col 2: Error 0.0 -> MSE=0.0 -> RMSE=0.0
    # MCRMSE = (1.0 + 2.0 + 0.0) / 3 = 1.0

    metric_targets = torch.zeros((10, 3))
    metric_preds = torch.zeros((10, 3))

    metric_preds[:, 0] = 1.0
    metric_preds[:, 1] = 2.0
    metric_preds[:, 2] = 0.0

    score = calculate_mcrmse(metric_targets, metric_preds)
    print(f"  - Calculated MCRMSE: {score}")

    assert np.isclose(score, 1.0, atol=1e-6), f"Expected MCRMSE 1.0, got {score}"

    print("  - MCRMSE calculation verified.")

    # 6. Full Training Loop Demonstration
    # ----------------------------------------------------------------
    print("\n[5/5] Running Training Loop (Debug Mode)...")

    # run_training with debug=True runs for 2 epochs on a subset of 100 samples
    # It handles data loading, model init, training, validation, and saving.
    best_model_path = run_training(
        debug=True,
        epochs=1,
        batch_size=4,
        load_cached_data=True,  # Use the cache we generated in step 2 if compatible, or re-gen
    )

    print(f"  - Training finished. Best model path: {best_model_path}")

    assert os.path.exists(best_model_path), "Best model file was not saved."

    # Verify we can load the saved model
    loaded_model = RNAModel(Config)
    loaded_model.load_state_dict(torch.load(best_model_path, map_location="cpu"))
    print("  - Saved model loaded successfully.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
