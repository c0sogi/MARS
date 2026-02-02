import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.dataset import SETIDataset
from library.model import SiameseGatedEfficientNet
from library.engine import train_one_epoch, validate_one_epoch, fit


def main():
    print("Initializing SETI Library Demo...")

    # --- 1. Configuration & Setup ---
    # Override Config for a fast demonstration (Debug Mode)
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 64  # Small subset for speed
    Config.BATCH_SIZE = 8
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo stability
    Config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Ensure reproducibility
    seed_everything(Config.SEED)

    print(f"Configuration:")
    print(f"  Device: {Config.DEVICE}")
    print(f"  Debug Mode: {Config.DEBUG}")
    print(f"  Sample Size: {Config.DEBUG_SAMPLE_SIZE}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")

    # --- 2. Dataset Logic Verification ---
    print("\n--- Verifying Dataset Logic ---")

    # Load metadata (Assumes metadata generation script has run)
    if not os.path.exists(Config.TRAIN_CSV):
        raise FileNotFoundError(f"Metadata file not found: {Config.TRAIN_CSV}")

    df_train = pd.read_csv(Config.TRAIN_CSV)

    # Initialize Dataset in Train mode
    train_dataset = SETIDataset(df_train, mode="train")

    # Assert debug sizing works
    assert (
        len(train_dataset) == Config.DEBUG_SAMPLE_SIZE
    ), f"Dataset length {len(train_dataset)} does not match debug size {Config.DEBUG_SAMPLE_SIZE}"

    # Fetch a single sample
    on_tensor, off_tensor, target = train_dataset[0]

    # Verify Shapes
    # Expected: (3, 288, 256) based on Config.INPUT_SIZE=(288, 256)
    expected_h, expected_w = Config.INPUT_SIZE
    expected_shape = (3, expected_h, expected_w)

    print(
        f"Sample shapes: On={on_tensor.shape}, Off={off_tensor.shape}, Target={target}"
    )

    if on_tensor.shape != expected_shape:
        raise AssertionError(
            f"On-target tensor shape mismatch. Got {on_tensor.shape}, expected {expected_shape}"
        )
    if off_tensor.shape != expected_shape:
        raise AssertionError(
            f"Off-target tensor shape mismatch. Got {off_tensor.shape}, expected {expected_shape}"
        )
    if not isinstance(target, torch.Tensor):
        raise AssertionError("Target is not a torch.Tensor")

    print("Dataset verification passed.")

    # --- 3. Model Initialization & Forward Pass ---
    print("\n--- Verifying Model Architecture ---")

    # Initialize model (pretrained=False to avoid downloading weights during demo)
    model = SiameseGatedEfficientNet(
        backbone_name=Config.BACKBONE_NAME, pretrained=False
    )
    model.to(Config.DEVICE)

    # Create dummy batch matching dataset output
    dummy_on = torch.randn(Config.BATCH_SIZE, 3, expected_h, expected_w).to(
        Config.DEVICE
    )
    dummy_off = torch.randn(Config.BATCH_SIZE, 3, expected_h, expected_w).to(
        Config.DEVICE
    )

    # Perform forward pass
    logits = model(dummy_on, dummy_off)

    print(f"Logits shape: {logits.shape}")

    # Assert output shape is (Batch_Size, 1)
    if logits.shape != (Config.BATCH_SIZE, 1):
        raise AssertionError(
            f"Model output shape mismatch. Got {logits.shape}, expected {(Config.BATCH_SIZE, 1)}"
        )

    print("Model forward pass verification passed.")

    # --- 4. Training Engine Demonstration ---
    print("\n--- Verifying Training Engine ---")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0
    )

    # Create Validation Dataset/Loader (using train df for demo convenience)
    val_dataset = SETIDataset(df_train, mode="val")
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Setup Optimizer and Criterion
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # Test train_one_epoch
    print("Running single training epoch...")
    train_loss = train_one_epoch(
        model, train_loader, criterion, optimizer, Config.DEVICE, epoch=0
    )
    print(f"Train Loss: {train_loss:.4f}")

    if np.isnan(train_loss):
        raise AssertionError("Training loss returned NaN.")

    # Test validate_one_epoch
    print("Running single validation epoch...")
    val_loss, val_auc = validate_one_epoch(model, val_loader, criterion, Config.DEVICE)
    print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

    if np.isnan(val_loss):
        raise AssertionError("Validation loss returned NaN.")

    # --- 5. Full Fit Loop Demonstration ---
    print("\n--- Verifying Full Fit Loop ---")

    # Define a temporary save path
    save_path = os.path.join(Config.WORKING_DIR, "demo_model.pth")

    # Run fit()
    best_auc = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,  # No scheduler for this short demo
        criterion=criterion,
        device=Config.DEVICE,
        epochs=1,  # Only 1 epoch
        patience=1,
        save_path=save_path,
    )

    print(f"Fit loop completed. Best AUC: {best_auc:.4f}")

    # Verify model checkpoint was saved
    if not os.path.exists(save_path):
        raise AssertionError(f"Model checkpoint not found at {save_path}")

    print("\nAll demonstrations and verifications completed successfully.")


if __name__ == "__main__":
    main()
