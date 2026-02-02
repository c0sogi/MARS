import os
import sys
import torch
import numpy as np

# Import library components
from library.config import Config
from library.utils import seed_everything, compute_metric
from library.data import get_dataloaders
from library.model import VentilatorModel
from library.train import Trainer


def main():
    print("=== Ventilator Pressure Prediction: Library Usage Demo ===")

    # ---------------------------------------------------------
    # 1. Configuration Setup (Optimized for Speed)
    # ---------------------------------------------------------
    print("\n[Step 1] Configuring environment for fast execution...")

    # Modify Config attributes for a quick debug run
    Config.DEBUG = True
    Config.DEBUG_BREATHS = 200  # Use only 200 breaths for speed
    Config.EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 16  # Small batch size for small data
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution

    # Use a specific demo directory to avoid clutter
    Config.WORK_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORK_DIR, "cache")
    Config.MODEL_PATH = os.path.join(Config.WORK_DIR, "model.pth")

    # Initialize directories
    Config.setup()

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    print("Configuration updated. Debug mode enabled.")

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("\n[Step 2] Loading DataLoaders...")

    # We pass debug=True and load_cached_data=False to ensure we generate
    # the small debug subset fresh and don't load a full cached dataset.
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False,
        debug=True,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # Validation: Check if loaders are populated
    assert len(train_loader) > 0, "Train loader should not be empty."
    assert len(val_loader) > 0, "Validation loader should not be empty."

    # Fetch a single batch to inspect structure
    batch = next(iter(train_loader))
    x, y, u_out = batch["x"], batch["y"], batch["u_out"]

    print(f"Batch Loaded - X shape: {x.shape}, Y shape: {y.shape}")

    # Validation: Check tensor shapes
    # Expected X: (Batch, 80, Features)
    # Expected Y: (Batch, 80)
    assert x.ndim == 3, f"Input X should be 3D, got {x.ndim}"
    assert x.shape[1] == 80, f"Sequence length should be 80, got {x.shape[1]}"
    assert y.shape == (Config.BATCH_SIZE, 80), f"Target Y shape mismatch: {y.shape}"
    assert u_out.shape == (
        Config.BATCH_SIZE,
        80,
    ), f"u_out shape mismatch: {u_out.shape}"

    # ---------------------------------------------------------
    # 3. Model Initialization & Forward Pass
    # ---------------------------------------------------------
    print("\n[Step 3] Initializing Model and running Forward Pass...")

    model = VentilatorModel()
    model.to(Config.DEVICE)

    # Move inputs to device
    x_dev = x.to(Config.DEVICE)

    # Run forward pass
    final_out, aux_out = model(x_dev)

    print(f"Model Output Shapes - Final: {final_out.shape}, Aux: {aux_out.shape}")

    # Validation: Check output shapes
    # Model returns (Batch, Seq, 1)
    expected_out_shape = (Config.BATCH_SIZE, 80, 1)
    assert (
        final_out.shape == expected_out_shape
    ), f"Final output shape mismatch: {final_out.shape}"
    assert (
        aux_out.shape == expected_out_shape
    ), f"Aux output shape mismatch: {aux_out.shape}"

    # ---------------------------------------------------------
    # 4. Metric Utility Test
    # ---------------------------------------------------------
    print("\n[Step 4] Testing Metric Calculation...")

    # Prepare tensors for metric (move to CPU and detach)
    preds = final_out.squeeze(-1).detach().cpu()
    targets = y.cpu()
    u_out_cpu = u_out.cpu()

    # Calculate MAE
    mae = compute_metric(preds, targets, u_out_cpu)
    print(f"Calculated MAE (Untrained Model): {mae:.4f}")

    # Validation: MAE should be a non-negative float
    assert isinstance(mae, float), "Metric should return a float"
    assert mae >= 0, "MAE cannot be negative"

    # ---------------------------------------------------------
    # 5. Training Loop Execution
    # ---------------------------------------------------------
    print("\n[Step 5] Starting Training Loop...")

    trainer = Trainer(model, train_loader, val_loader)

    # Run training (Config.EPOCHS is set to 2)
    trainer.fit()

    # Validation: Check if model file was saved
    if os.path.exists(Config.MODEL_PATH):
        print(f"\nSuccess: Model saved to {Config.MODEL_PATH}")
    else:
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_PATH}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
