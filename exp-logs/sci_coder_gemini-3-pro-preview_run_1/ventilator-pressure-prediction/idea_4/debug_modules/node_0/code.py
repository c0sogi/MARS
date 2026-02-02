import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil
from torch.utils.data import DataLoader

# Ensure the library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything
from library.dataset import prepare_data
from library.model import PhysicsResidualNet
from library.train import train_one_epoch, valid_one_epoch, masked_mae_loss


def main():
    print("=== Ventilator Pressure Prediction: Library Demo ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup
    # -------------------------------------------------------------------------
    print("[1] Configuring environment for rapid demonstration...")

    # Set seed for reproducibility
    seed_everything(42)

    # Override Config attributes for a fast, lightweight run
    Config.debug = True  # Uses only 200 breaths (~16k rows) instead of full dataset
    Config.exp_name = "demo_execution"

    # Update paths to use a dedicated demo directory
    Config.working_dir = os.path.join("./working", Config.exp_name)
    Config.cache_dir = Config.working_dir
    Config.model_path = os.path.join(Config.working_dir, "model.pth")
    Config.submission_path = os.path.join(Config.working_dir, "submission.csv")

    # Reduce Model Complexity for Speed
    Config.lstm_layers = 1
    Config.lstm_hidden_size = 64
    Config.cnn_filters = 16
    Config.cnn_kernel_sizes = [3, 5]  # Reduced kernels
    Config.lstm_input_size = Config.cnn_filters * len(Config.cnn_kernel_sizes)

    # Reduce Training Parameters
    Config.epochs = 2
    Config.train_batch_size = 16
    Config.val_batch_size = 32
    Config.num_workers = 0  # Avoid multiprocessing overhead for small demo

    # Clean previous demo run if exists
    if os.path.exists(Config.working_dir):
        shutil.rmtree(Config.working_dir)
    os.makedirs(Config.working_dir, exist_ok=True)

    print(f"    Debug Mode: {Config.debug}")
    print(f"    Working Directory: {Config.working_dir}")
    print(f"    Device: {Config.device}")

    # -------------------------------------------------------------------------
    # 2. Data Preparation Pipeline
    # -------------------------------------------------------------------------
    print("\n[2] Executing Data Preparation Pipeline...")

    # Prepare Training Data (computes features, fits scaler, caches result)
    print("    Processing Training Data...")
    train_dataset = prepare_data("train", load_cached_data=False)

    # Prepare Validation Data (computes features, loads scaler, caches result)
    print("    Processing Validation Data...")
    val_dataset = prepare_data("val", load_cached_data=False)

    # Verify Dataset Integrity
    print("    Verifying Dataset structures...")
    sample = train_dataset[0]

    # Check Feature Shape: (Seq_Len, Input_Dim)
    assert sample["x"].shape == (
        Config.seq_len,
        Config.input_dim,
    ), f"Shape mismatch! Expected ({Config.seq_len}, {Config.input_dim}), got {sample['x'].shape}"

    # Check Target Shape: (Seq_Len,)
    assert sample["y"].shape == (
        Config.seq_len,
    ), f"Target shape mismatch! Expected ({Config.seq_len},), got {sample['y'].shape}"

    # Check Physics Feature (Theoretical Pressure)
    assert sample["p_theory"].shape == (
        Config.seq_len,
    ), "Theoretical pressure shape mismatch."

    print(f"    Train Dataset Size: {len(train_dataset)} breaths")
    print(f"    Val Dataset Size: {len(val_dataset)} breaths")
    print("    Dataset verification passed.")

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("\n[3] Initializing Physics-Residual Model...")

    model = PhysicsResidualNet()
    model.to(Config.device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Model instantiated with {num_params:,} trainable parameters.")

    # -------------------------------------------------------------------------
    # 4. Forward Pass & Loss Calculation
    # -------------------------------------------------------------------------
    print("\n[4] Testing Forward Pass and Loss Logic...")

    # Create DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=Config.train_batch_size, shuffle=True, num_workers=0
    )

    # Get a batch
    batch = next(iter(train_loader))
    x = batch["x"].to(Config.device)
    u_out = batch["u_out"].to(Config.device)
    p_theory = batch["p_theory"].to(Config.device)
    y = batch["y"].to(Config.device)

    # Forward Pass
    model.train()
    preds = model(x, p_theory)

    # Verify Output Shape: (Batch, Seq_Len)
    assert preds.shape == (
        x.size(0),
        Config.seq_len,
    ), f"Prediction shape mismatch. Expected ({x.size(0)}, {Config.seq_len}), got {preds.shape}"

    # Calculate Loss
    loss = masked_mae_loss(preds, y, u_out)

    print(f"    Batch Loss: {loss.item():.6f}")
    assert not torch.isnan(loss), "Loss is NaN!"
    assert loss.item() >= 0, "Loss cannot be negative."
    print("    Forward pass and loss calculation verified.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n[5] Running Short Training Loop...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=Config.epochs
    )

    for epoch in range(Config.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, Config.device
        )

        # Use a small subset of val for speed in this demo
        val_loader = DataLoader(
            val_dataset, batch_size=Config.val_batch_size, shuffle=False
        )
        val_loss = valid_one_epoch(model, val_loader, Config.device)

        print(
            f"    Epoch {epoch+1}/{Config.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        assert train_loss < 1000, "Train loss abnormally high."
        assert val_loss < 1000, "Val loss abnormally high."

    print("    Training loop executed successfully.")

    # -------------------------------------------------------------------------
    # 6. Artifact Management (Save/Load)
    # -------------------------------------------------------------------------
    print("\n[6] Verifying Model Checkpointing...")

    # Save
    torch.save(model.state_dict(), Config.model_path)
    if not os.path.exists(Config.model_path):
        raise FileNotFoundError("Failed to save model file.")
    print(f"    Model saved to {Config.model_path}")

    # Load
    loaded_model = PhysicsResidualNet()
    loaded_model.load_state_dict(
        torch.load(Config.model_path, map_location=Config.device)
    )
    loaded_model.to(Config.device)
    print("    Model loaded successfully.")

    # Verify loaded model predictions match original
    loaded_model.eval()
    with torch.no_grad():
        loaded_preds = loaded_model(x, p_theory)

    # Allow small floating point divergence
    # Note: preds was from model.train(), loaded is eval(), so dropout might cause diffs if we don't set original to eval
    model.eval()
    with torch.no_grad():
        eval_preds = model(x, p_theory)

    diff_eval = torch.abs(eval_preds - loaded_preds).max().item()
    print(f"    Prediction difference after reload: {diff_eval:.8f}")
    assert diff_eval < 1e-6, "Loaded model predictions diverge from original."

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
