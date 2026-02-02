import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Ensure the current directory is in the path for library imports
sys.path.append(os.getcwd())

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, WeightedL1Loss, compute_metric
from library.dataset import prepare_data, add_features
from library.model import DFLB_BiLSTM
from library.train import Trainer


def run_demo():
    print("=== Starting Demonstration of Ventilator Pressure Prediction Pipeline ===")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("\n[Step 1] Configuring environment for demo...")

    # Override Config for speed and isolation
    Config.DEBUG = True
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 16
    Config.SCHEDULER_T_MAX = 2  # Match epochs for scheduler
    Config.WORKING_DIR = "./working/demo_verification"

    # Update derived paths in Config
    Config.CACHE_DIR = Config.WORKING_DIR
    Config.MODEL_CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.BEST_MODEL_PATH = os.path.join(Config.MODEL_CHECKPOINT_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Clean up previous demo run if exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)

    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.MODEL_CHECKPOINT_DIR, exist_ok=True)

    # Set seed
    set_seed(Config.SEED)
    print("Configuration updated. Debug mode: ON.")

    # ==========================================
    # 2. Verify Logic: Metrics
    # ==========================================
    print("\n[Step 2] Verifying Metric Logic...")

    # Create dummy data
    # 2 samples. Sample 1: Insp (u_out=0), Error=2. Sample 2: Exp (u_out=1), Error=2.
    preds = torch.tensor([10.0, 10.0])
    targets = torch.tensor([12.0, 12.0])
    u_out = torch.tensor([0.0, 1.0])

    # Test WeightedL1Loss
    # Weights: Insp=1.0, Exp=0.1
    # Loss = (|10-12|*1.0 + |10-12|*0.1) / 2 = (2.0 + 0.2) / 2 = 1.1
    criterion = WeightedL1Loss()
    loss = criterion(preds, targets, u_out)

    expected_loss = 1.1
    assert (
        abs(loss.item() - expected_loss) < 1e-6
    ), f"WeightedL1Loss failed. Expected {expected_loss}, got {loss.item()}"
    print("WeightedL1Loss logic verified.")

    # Test compute_metric (MAE on Inspiratory phase only)
    # Should only consider index 0. Error = 2.0.
    metric = compute_metric(preds, targets, u_out)
    expected_metric = 2.0
    assert (
        abs(metric - expected_metric) < 1e-6
    ), f"compute_metric failed. Expected {expected_metric}, got {metric}"
    print("compute_metric logic verified.")

    # ==========================================
    # 3. Verify Logic: Feature Engineering
    # ==========================================
    print("\n[Step 3] Verifying Feature Engineering...")

    # Create a dummy dataframe representing one breath (id=1)
    # time_step: 0.0 -> 0.1 -> 0.3
    # u_in: 10 -> 20 -> 0
    dummy_data = {
        "breath_id": [1, 1, 1],
        "time_step": [0.0, 0.1, 0.3],
        "u_in": [10.0, 20.0, 0.0],
        "u_out": [0, 0, 1],
        "R": [20, 20, 20],
        "C": [50, 50, 50],
        "pressure": [5, 6, 7],
        "id": [1, 2, 3],  # Needed for sorting
    }
    df_dummy = pd.DataFrame(dummy_data)

    df_processed = add_features(df_dummy)

    # Verify dt (time delta)
    # dt[0] = 0 (fillna), dt[1] = 0.1, dt[2] = 0.2
    expected_dt = [0.0, 0.1, 0.2]
    np.testing.assert_allclose(
        df_processed["dt"].values,
        expected_dt,
        atol=1e-6,
        err_msg="dt calculation wrong",
    )

    # Verify volume (cumulative sum of u_in * dt)
    # vol[0] = 10*0 = 0
    # vol[1] = 0 + 20*0.1 = 2.0
    # vol[2] = 2.0 + 0*0.2 = 2.0
    expected_vol = [0.0, 2.0, 2.0]
    np.testing.assert_allclose(
        df_processed["volume"].values,
        expected_vol,
        atol=1e-6,
        err_msg="volume calculation wrong",
    )

    print("Feature engineering logic verified.")

    # ==========================================
    # 4. Data Preparation
    # ==========================================
    print("\n[Step 4] Preparing Data (Debug Mode)...")
    # This will load raw data, filter for debug subset, process features, and create loaders
    train_loader, val_loader, test_loader = prepare_data(
        debug=True, load_cached_data=False
    )

    # Check loader batch shape
    sample_x, sample_y, sample_uout = next(iter(train_loader))
    print(f"Train Batch Shape - X: {sample_x.shape}, y: {sample_y.shape}")

    # Assert shapes
    # X: (Batch, Seq_Len, Features)
    # y: (Batch, Seq_Len)
    assert sample_x.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    assert sample_x.shape[1] == Config.MAX_SEQ_LEN, "Sequence length mismatch"
    assert sample_y.shape == (
        Config.BATCH_SIZE,
        Config.MAX_SEQ_LEN,
    ), "Target shape mismatch"

    input_dim = sample_x.shape[-1]
    print(f"Detected Input Dimension: {input_dim}")

    # ==========================================
    # 5. Model Initialization & Forward Pass
    # ==========================================
    print("\n[Step 5] Initializing Model...")
    device = torch.device(Config.DEVICE)
    model = DFLB_BiLSTM(input_dim=input_dim).to(device)

    # Verify Forward Pass
    dummy_input = torch.randn(2, Config.MAX_SEQ_LEN, input_dim).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (
        2,
        Config.MAX_SEQ_LEN,
    ), f"Model output shape mismatch. Expected (2, 80), got {output.shape}"
    print("Model forward pass verified.")

    # ==========================================
    # 6. Training Loop
    # ==========================================
    print("\n[Step 6] Running Training Loop...")
    trainer = Trainer(model, device)

    # Run fit
    trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    # Verify checkpoint creation
    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), "Best model checkpoint was not saved."
    print("Training loop completed and model saved.")

    # ==========================================
    # 7. Inference
    # ==========================================
    print("\n[Step 7] Running Inference...")
    predictions = trainer.predict(test_loader)

    # Verify predictions shape
    # In debug mode, test_loader has a small subset of breaths.
    # We need to calculate expected size.
    # prepare_data debug mode filters test breaths to first 20 breaths.
    # 20 breaths * 80 steps = 1600 predictions.
    expected_preds = 20 * 80
    assert (
        len(predictions) == expected_preds
    ), f"Prediction count mismatch. Expected {expected_preds}, got {len(predictions)}"

    print(f"Generated {len(predictions)} predictions.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
