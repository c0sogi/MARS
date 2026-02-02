import os
import sys
import torch
import torch.optim as optim
import numpy as np
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_data_loaders, get_test_loader
from library.model import PCGIBiLSTM
from library.train import WeightedL1Loss, train_epoch, validate_epoch


def main():
    print("=== Ventilator Pressure Prediction: Library Usage Demo ===")

    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")

    # Override Config for speed and debug mode
    Config.DEBUG = True
    Config.DEBUG_BREATHS = 100  # Use only 100 breaths for speed
    Config.EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 16  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Set a specific working directory for this demo
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.BEST_MODEL_PATH = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    Config.SCALER_PATH = os.path.join(Config.CACHE_DIR, "scaler_params.npz")

    # Initialize directories
    Config.setup()

    # Set random seeds
    seed_everything(Config.SEED)
    print("    Configuration complete. Debug mode enabled.")

    # ---------------------------------------------------------
    # 2. Data Loading & Processing
    # ---------------------------------------------------------
    print("\n[2] Testing Data Pipeline (Feature Engineering & Loading)...")

    # Force reload to demonstrate processing logic (load_cached_data=False)
    train_loader, val_loader = get_data_loaders(load_cached_data=False)

    # Verify Train Loader
    X_batch, u_out_batch, y_batch = next(iter(train_loader))

    print(
        f"    Batch Shapes -> X: {X_batch.shape}, u_out: {u_out_batch.shape}, y: {y_batch.shape}"
    )

    # Assertions
    # Shape: (Batch, Seq_Len=80, Features=14)
    assert X_batch.shape == (
        Config.BATCH_SIZE,
        80,
        14,
    ), f"Incorrect X shape: {X_batch.shape}"
    assert u_out_batch.shape == (
        Config.BATCH_SIZE,
        80,
    ), f"Incorrect u_out shape: {u_out_batch.shape}"
    assert y_batch.shape == (
        Config.BATCH_SIZE,
        80,
    ), f"Incorrect y shape: {y_batch.shape}"
    assert X_batch.dtype == torch.float32, "X should be float32"

    print("    Data Pipeline verification passed.")

    # ---------------------------------------------------------
    # 3. Model Initialization & Forward Pass
    # ---------------------------------------------------------
    print("\n[3] Testing Model Architecture (PC-GI-BiLSTM)...")

    device = Config.DEVICE
    model = PCGIBiLSTM().to(device)

    # Move batch to device
    X_batch = X_batch.to(device)
    u_out_batch = u_out_batch.to(device)
    y_batch = y_batch.to(device)

    # Forward pass
    preds = model(X_batch)

    print(f"    Prediction Shape: {preds.shape}")

    # Assertions
    assert preds.shape == (
        Config.BATCH_SIZE,
        80,
    ), f"Output shape mismatch: {preds.shape}"
    assert not torch.isnan(preds).any(), "Model output contains NaNs"
    assert preds.requires_grad, "Gradients not tracking on output"

    print("    Model forward pass verification passed.")

    # ---------------------------------------------------------
    # 4. Loss Function Verification
    # ---------------------------------------------------------
    print("\n[4] Testing Loss Function (Weighted L1)...")

    criterion = WeightedL1Loss().to(device)
    loss = criterion(preds, y_batch, u_out_batch)

    print(f"    Calculated Loss: {loss.item():.4f}")

    # Assertions
    assert loss.dim() == 0, "Loss should be a scalar"
    assert loss.item() >= 0, "Loss cannot be negative"

    print("    Loss function verification passed.")

    # ---------------------------------------------------------
    # 5. Training Loop Simulation
    # ---------------------------------------------------------
    print("\n[5] Simulating Training Loop...")

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Run 1 Training Epoch
    print("    Running Train Epoch 1...")
    train_metrics = train_epoch(
        model, train_loader, optimizer, criterion, device, epoch=1
    )
    print(f"    Train Metrics: {train_metrics}")

    assert "Loss" in train_metrics, "Train metrics missing Loss"

    # Run 1 Validation Epoch
    print("    Running Validation Epoch 1...")
    val_metrics = validate_epoch(model, val_loader, criterion, device)
    print(f"    Val Metrics:   {val_metrics}")

    assert "MAE" in val_metrics, "Val metrics missing MAE"

    print("    Training loop simulation passed.")

    # ---------------------------------------------------------
    # 6. Inference Pipeline
    # ---------------------------------------------------------
    print("\n[6] Testing Inference Pipeline...")

    test_loader = get_test_loader(load_cached_data=False)
    X_test, u_out_test = next(iter(test_loader))

    # Move to device
    X_test = X_test.to(device)

    model.eval()
    with torch.no_grad():
        test_preds = model(X_test)

    print(f"    Test Batch Prediction Shape: {test_preds.shape}")
    assert test_preds.shape == (Config.BATCH_SIZE, 80), "Test prediction shape mismatch"

    print("    Inference pipeline passed.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
