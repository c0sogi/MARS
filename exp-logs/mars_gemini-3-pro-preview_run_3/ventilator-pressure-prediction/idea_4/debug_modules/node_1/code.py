import os
import torch
import numpy as np
import pandas as pd
import shutil
import time

# Import library components
from library.config import Config, set_seed
from library.utils import get_device
from library.features import get_all_feature_names
from library.dataset import DataManager, VentilatorDataset
from library.model import DisentangledTCNLSTM
from library.engine import MaskedMAELoss, train_fn, eval_fn


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup & Configuration Overrides for Speed
    # We create a separate working directory for this demo to avoid conflicts.
    # We enable DEBUG mode to drastically reduce dataset size (e.g., 200 breaths).
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)

    print(f"Configuring environment in {demo_dir}...")

    # Update Config paths dynamically
    Config.WORKING_DIR = demo_dir
    Config.TRAIN_CACHE_X = os.path.join(demo_dir, "train_x.npy")
    Config.TRAIN_CACHE_Y = os.path.join(demo_dir, "train_y.npy")
    Config.VAL_CACHE_X = os.path.join(demo_dir, "val_x.npy")
    Config.VAL_CACHE_Y = os.path.join(demo_dir, "val_y.npy")
    Config.TEST_CACHE_X = os.path.join(demo_dir, "test_x.npy")
    Config.SCALER_PATH = os.path.join(demo_dir, "scaler_stats.npz")
    Config.MODEL_PATH = os.path.join(demo_dir, "model.pth")
    Config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Hyperparameters for fast execution
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 200  # Only use 200 breaths
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Initialize directories and seed
    Config.setup()
    set_seed(Config.SEED)
    device = get_device()
    print(f"Device selected: {device}")

    # 2. Data Pipeline Verification
    print("\n--- Verifying Data Pipeline ---")
    dm = DataManager()

    # Load Training Data
    # Note: The first run will process the CSVs. Subsequent runs would use cache.
    # In DEBUG mode, DataManager loads the file but truncates X/y in memory.
    print("Loading training dataloader...")
    train_loader = dm.get_dataloader(
        "train", batch_size=Config.BATCH_SIZE, shuffle=True
    )

    # Fetch one batch to verify structure
    batch_inputs, batch_targets = next(iter(train_loader))

    # Assertions
    # TCN Input: (Batch, Channels, Length=80)
    # LSTM Input: (Batch, Length=80, Channels)
    # Skip Input: (Batch, Length=80, Channels)
    # Targets: (Batch, Length=80)

    print("Verifying batch shapes...")
    tcn_shape = batch_inputs["tcn"].shape
    lstm_shape = batch_inputs["lstm"].shape
    skip_shape = batch_inputs["skip"].shape
    target_shape = batch_targets.shape

    input_dims = Config.get_input_dims()

    assert tcn_shape == (
        Config.BATCH_SIZE,
        input_dims["tcn"],
        80,
    ), f"TCN shape mismatch: {tcn_shape}"
    assert lstm_shape == (
        Config.BATCH_SIZE,
        80,
        input_dims["lstm"],
    ), f"LSTM shape mismatch: {lstm_shape}"
    assert skip_shape == (
        Config.BATCH_SIZE,
        80,
        input_dims["skip"],
    ), f"Skip shape mismatch: {skip_shape}"
    assert target_shape == (
        Config.BATCH_SIZE,
        80,
    ), f"Target shape mismatch: {target_shape}"

    print("Data shapes verified successfully.")

    # 3. Model Architecture Verification
    print("\n--- Verifying Model Architecture ---")
    model = DisentangledTCNLSTM().to(device)

    # Move batch to device
    batch_inputs_dev = {k: v.to(device) for k, v in batch_inputs.items()}

    # Forward pass
    with torch.no_grad():
        preds = model(batch_inputs_dev)

    print(f"Model output shape: {preds.shape}")
    assert preds.shape == (
        Config.BATCH_SIZE,
        80,
    ), f"Model output shape mismatch. Expected ({Config.BATCH_SIZE}, 80), got {preds.shape}"

    print("Model forward pass successful.")

    # 4. Loss Function Logic Verification
    print("\n--- Verifying Loss Function (Masked MAE) ---")
    loss_fn = MaskedMAELoss()

    # Create synthetic data
    # Case: 2 samples, 3 time steps
    # Sample 1: u_out=[0, 0, 1] (Inspiratory, Inspiratory, Expiratory)
    # Sample 2: u_out=[0, 1, 1] (Inspiratory, Expiratory, Expiratory)

    syn_pred = torch.tensor([[10.0, 12.0, 5.0], [20.0, 5.0, 5.0]], dtype=torch.float32)

    syn_target = torch.tensor(
        [
            [12.0, 12.0, 100.0],  # Errors: |10-12|=2, |12-12|=0, Ignored
            [24.0, 100.0, 100.0],  # Errors: |20-24|=4, Ignored, Ignored
        ],
        dtype=torch.float32,
    )

    syn_u_out = torch.tensor([[0, 0, 1], [0, 1, 1]], dtype=torch.float32)

    # Manual Calculation
    # Valid points: (0,0), (0,1), (1,0) -> Total 3 points
    # Errors: 2.0, 0.0, 4.0
    # Mean: (2+0+4) / 3 = 2.0

    calculated_loss = loss_fn(syn_pred, syn_target, syn_u_out).item()
    print(f"Calculated Loss: {calculated_loss}")

    assert (
        abs(calculated_loss - 2.0) < 1e-6
    ), f"Loss calculation incorrect. Expected 2.0, got {calculated_loss}"

    print("Loss function logic verified.")

    # 5. Training Loop Demonstration
    print("\n--- Running Training Loop (1 Epoch) ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Using the train_fn from library.engine
    # Note: train_fn expects the dataloader to yield (inputs, targets)
    train_loss = train_fn(model, train_loader, optimizer, device)
    print(f"Training Epoch Completed. Loss: {train_loss:.4f}")

    # Verify weights updated (simple check: save initial state, compare?)
    # Or just rely on function completion.

    print("\n--- Running Evaluation ---")
    val_loader = dm.get_dataloader(
        "validation", batch_size=Config.BATCH_SIZE, shuffle=False
    )
    val_loss = eval_fn(model, val_loader, device)
    print(f"Validation Completed. Loss: {val_loss:.4f}")

    # 6. Submission Generation Test
    print("\n--- Verifying Submission Generation ---")
    # We will simulate the prediction step on a subset of test data
    test_loader = dm.get_dataloader("test", batch_size=Config.BATCH_SIZE, shuffle=False)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            batch_preds = model(inputs)
            all_preds.append(batch_preds.cpu().numpy())

    predictions = np.concatenate(all_preds, axis=0)
    flat_preds = predictions.flatten()

    # Verify count
    # In DEBUG mode, we have Config.DEBUG_SAMPLE_SIZE breaths * 80 steps
    expected_rows = Config.DEBUG_SAMPLE_SIZE * 80
    assert (
        len(flat_preds) == expected_rows
    ), f"Prediction count mismatch. Expected {expected_rows}, got {len(flat_preds)}"

    # Generate dummy submission file
    # We need to load the test csv to get IDs, but for this demo we'll just create a dummy ID list
    # matching the length, as loading the full test.csv is slow.
    dummy_ids = np.arange(1, len(flat_preds) + 1)
    submission = pd.DataFrame({"id": dummy_ids, "pressure": flat_preds})

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission file generated at {Config.SUBMISSION_PATH}")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
