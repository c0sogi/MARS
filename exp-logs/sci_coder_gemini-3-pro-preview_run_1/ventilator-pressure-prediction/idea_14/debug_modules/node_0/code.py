import os
import shutil
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything, compute_metric
from library.data import get_dataloaders
from library.model import HighCapacityCompositeModel
from library.train import Trainer, MaskedL1Loss


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Reproducibility
    seed_everything(42)

    # 2. Configure for Demo/Speed
    # We modify the Config class attributes directly to run a lightweight version
    print("Configuring environment for rapid demonstration...")
    Config.debug = True  # Use small subset of data
    Config.epochs = 2  # Run only 2 epochs
    Config.batch_size = 16  # Small batch size
    Config.num_workers = 2  # Reduce worker overhead for small data
    Config.exp_name = "demo_execution"
    Config.working_dir = os.path.join("./working", Config.exp_name)
    Config.cache_dir = os.path.join(Config.working_dir, "cache")
    Config.model_path = os.path.join(Config.working_dir, "model.pth")

    # Re-run setup to create new directories
    Config.setup()
    print(f"Working directory set to: {Config.working_dir}")

    # 3. Data Pipeline Verification
    print("\n--- Testing Data Pipeline ---")
    # Force reload (load_cached_data=False) to test feature engineering logic
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=Config.debug,
        batch_size=Config.batch_size,
        num_workers=Config.num_workers,
        load_cached_data=False,
    )

    # Fetch a single batch to verify shapes and content
    batch = next(iter(train_loader))
    x, y, u_out = batch["x"], batch["y"], batch["u_out"]

    print(f"Batch Shapes -> X: {x.shape}, Y: {y.shape}, u_out: {u_out.shape}")

    # Assertions
    # Shape: (Batch, Time, Features)
    assert x.ndim == 3, "Input X must be 3D tensor"
    assert x.shape[1] == 80, "Time dimension must be 80"
    assert x.shape[2] == len(
        Config.features
    ), f"Feature count mismatch. Expected {len(Config.features)}, got {x.shape[2]}"

    # Shape: (Batch, Time)
    assert y.ndim == 2, "Target Y must be 2D tensor"
    assert u_out.ndim == 2, "Control u_out must be 2D tensor"

    # Value checks
    assert torch.all((u_out == 0) | (u_out == 1)), "u_out must be binary (0 or 1)"
    print("Data Pipeline verification passed.")

    # 4. Model Architecture Verification
    print("\n--- Testing Model Architecture ---")
    model = HighCapacityCompositeModel(config=Config)
    model.to(Config.device)

    # Move batch to device
    x_gpu = x.to(Config.device)

    # Forward pass
    final_pred, aux_pred = model(x_gpu)

    print(f"Output Shapes -> Final: {final_pred.shape}, Aux: {aux_pred.shape}")

    assert final_pred.shape == y.shape, "Final prediction shape mismatch"
    assert aux_pred.shape == y.shape, "Auxiliary prediction shape mismatch"
    print("Model Architecture verification passed.")

    # 5. Loss Function Logic Verification
    print("\n--- Testing Loss Function Logic ---")
    criterion = MaskedL1Loss(aux_weight=0.5)

    # Create synthetic data
    # Batch size 1, Time steps 2
    # Step 0: u_out=0 (Inspiration) -> Should count towards loss
    # Step 1: u_out=1 (Expiration)  -> Should be ignored

    pred_dummy = torch.tensor([[10.0, 20.0]], device=Config.device)
    target_dummy = torch.tensor([[12.0, 100.0]], device=Config.device)
    u_out_dummy = torch.tensor([[0.0, 1.0]], device=Config.device)

    # Expected Calculation:
    # Step 0: |10 - 12| = 2.0
    # Step 1: Ignored (masked)
    # Mean Loss = 2.0

    loss = criterion(pred_dummy, None, target_dummy, u_out_dummy)
    print(f"Calculated Loss: {loss.item()}")

    assert (
        abs(loss.item() - 2.0) < 1e-5
    ), f"Loss calculation incorrect. Expected 2.0, got {loss.item()}"
    print("Loss Function logic verification passed.")

    # 6. Training Loop Execution
    print("\n--- Testing Training Loop (2 Epochs) ---")
    trainer = Trainer(model, train_loader, val_loader, config=Config)

    # Run training
    trainer.fit()

    # Check artifacts
    if os.path.exists(Config.model_path):
        print(f"Model successfully saved to {Config.model_path}")
    else:
        raise FileNotFoundError("Model file was not generated.")

    # 7. Metric Verification
    print("\n--- Testing Metric Computation ---")
    model.eval()
    with torch.no_grad():
        # Predict on the sample batch
        preds, _ = model(x_gpu)

    mae = compute_metric(preds, y.to(Config.device), u_out.to(Config.device))
    print(f"Sample Batch MAE: {mae:.4f}")

    assert isinstance(mae, float), "Metric must return a float"
    assert mae >= 0, "MAE cannot be negative"

    print("\n=== All Demonstrations Completed Successfully ===")


if __name__ == "__main__":
    main()
