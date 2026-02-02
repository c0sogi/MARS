import os
import shutil
import torch
import torch.optim as optim
import numpy as np

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.data import get_loader
from library.model import DIN_CG_BiGRU
from library.loss import MCRMSELoss
from library.train import train_one_epoch, validate


def main():
    print("=== Starting Demo Execution ===\n")

    # 1. Override Configuration for Fast Demo Execution
    # We modify the Config class attributes directly to effect changes across all modules
    print("Configuring Demo Parameters...")
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Enable Debug mode to use a tiny subset of data
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 32

    # Reduce training parameters for speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Re-run setup to create the new directories
    Config.setup()

    # Set seed for reproducibility
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading Verification
    print("\n--- Testing Data Loading ---")
    # Initialize Train Loader
    train_loader = get_loader("train", batch_size=Config.BATCH_SIZE, shuffle=True)

    # Fetch one batch
    batch = next(iter(train_loader))
    features = batch["features"].to(device)
    bpps_indices = batch["bpps_indices"].to(device)
    bpps_mask = batch["bpps_mask"].to(device)
    targets = batch["targets"].to(device)

    print(f"Batch loaded. Batch size: {features.size(0)}")

    # Verify Shapes
    # Features: (B, Seq_Len=107, Channels=14)
    assert features.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.INPUT_DIM,
    ), f"Incorrect features shape: {features.shape}"

    # Targets: (B, Pred_Len=68, Num_Targets=5)
    # Note: Targets are only provided for the first 68 positions
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.PRED_LEN,
        Config.NUM_TARGETS,
    ), f"Incorrect targets shape: {targets.shape}"

    # BPPS Indices: (B, Seq_Len=107)
    assert bpps_indices.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), f"Incorrect bpps_indices shape: {bpps_indices.shape}"

    print("Data shapes verified successfully.")

    # 3. Model Initialization & Forward Pass
    print("\n--- Testing Model Forward Pass ---")
    model = DIN_CG_BiGRU().to(device)

    # Forward pass
    outputs = model(features, bpps_indices, bpps_mask)

    # Verify Output Shape
    # The model outputs predictions for the full sequence length (107)
    # Shape: (B, Seq_Len=107, Num_Targets=5)
    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.NUM_TARGETS,
    ), f"Incorrect output shape: {outputs.shape}"

    print(f"Model output shape: {outputs.shape}")
    print("Forward pass successful.")

    # 4. Loss Calculation Verification
    print("\n--- Testing Loss Calculation ---")
    criterion = MCRMSELoss()

    # Calculate loss
    loss = criterion(outputs, targets)

    # Check loss validity
    assert torch.isfinite(loss), "Loss is NaN or Infinite"
    assert loss.item() >= 0, "Loss is negative"

    print(f"Calculated MCRMSE Loss: {loss.item():.6f}")

    # 5. Training Loop Demonstration
    print("\n--- Testing Training Loop (2 Epochs) ---")

    # Setup Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # Load Validation Set
    val_loader = get_loader("val", batch_size=Config.BATCH_SIZE, shuffle=False)

    best_score = float("inf")

    for epoch in range(Config.EPOCHS):
        # Train Step
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validation Step
        val_score = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        print(
            f"Epoch {epoch + 1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score:.6f}"
        )

        # Simple checkpoint logic check
        if val_score < best_score:
            best_score = val_score
            # We won't actually save to disk in this demo to save time/space,
            # but this confirms the logic works.
            pass

    print(f"Training demo complete. Best Val Score: {best_score:.6f}")

    # 6. Cleanup
    print("\n--- Cleanup ---")
    if os.path.exists(Config.WORKING_DIR):
        try:
            shutil.rmtree(Config.WORKING_DIR)
            print(f"Removed temporary directory: {Config.WORKING_DIR}")
        except Exception as e:
            print(f"Warning: Could not remove temporary directory: {e}")

    print("\n=== Demo Execution Finished Successfully ===")


if __name__ == "__main__":
    main()
