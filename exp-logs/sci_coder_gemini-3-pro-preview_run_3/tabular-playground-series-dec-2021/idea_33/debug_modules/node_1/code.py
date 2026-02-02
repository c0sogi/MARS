import os
import torch
import pandas as pd
import numpy as np

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, set_performance_mode
from library.data import get_dataloaders
from library.model import ZeroInitDeepAsymmetricNet
from library.train import Trainer, train_pipeline


def main():
    print("=== Starting Library Usage Demonstration ===")

    # 1. Setup Environment
    # Ensure reproducibility and optimize performance settings
    print("\n[1] Setting up environment...")
    seed_everything(Config.SEED)
    set_performance_mode(deterministic=False, benchmark=True)
    print("    Random seeds set and CuDNN configured.")

    # 2. Data Loading Demonstration
    # We use debug_size to load a small subset for quick verification
    print("\n[2] Demonstrating Data Loading (with debug_size=1000)...")
    batch_size = 128
    debug_size = 1000

    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        batch_size=batch_size,
        load_cached_data=True,  # Will use cache if available, or process from metadata
        debug_size=debug_size,
        num_workers=2,
    )

    # Verify DataLoaders
    X_batch, y_batch = next(iter(train_loader))
    print(f"    Train Batch Shape: X={X_batch.shape}, y={y_batch.shape}")

    # Assertions to ensure data integrity
    assert X_batch.shape[0] <= batch_size, "Batch size exceeds limit"
    assert len(y_batch.shape) == 1, "Target should be 1D tensor"
    assert isinstance(X_batch, torch.Tensor), "Features must be a Tensor"
    assert isinstance(y_batch, torch.Tensor), "Targets must be a Tensor"

    input_dim = X_batch.shape[1]
    print(f"    Input Feature Dimension: {input_dim}")
    print("    Data loading verification passed.")

    # 3. Model Architecture Demonstration
    print("\n[3] Demonstrating Model Initialization and Forward Pass...")
    device = Config.DEVICE

    model = ZeroInitDeepAsymmetricNet(
        input_dim=input_dim,
        hidden_dim=Config.HIDDEN_DIM,
        num_blocks=2,  # Reduced for demo
        dcn_layers=1,  # Reduced for demo
        num_classes=Config.NUM_CLASSES,
        dropout=Config.DROPOUT,
    ).to(device)

    # Move batch to device
    X_batch = X_batch.to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(X_batch)

    print(f"    Output Logits Shape: {logits.shape}")

    # Assertions for model output
    assert logits.shape == (
        X_batch.shape[0],
        Config.NUM_CLASSES,
    ), f"Expected output shape {(X_batch.shape[0], Config.NUM_CLASSES)}, got {logits.shape}"
    print("    Model forward pass verification passed.")

    # 4. Trainer Class Demonstration
    print("\n[4] Demonstrating Trainer Class (Single Epoch)...")
    trainer = Trainer(model, device, Config)

    # Train for one epoch on the debug subset
    train_loss, train_acc = trainer.train_one_epoch(train_loader)
    val_loss, val_acc = trainer.validate(val_loader)

    print(f"    Debug Epoch - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"    Debug Epoch - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    assert not np.isnan(train_loss), "Training loss returned NaN"
    print("    Trainer class verification passed.")

    # 5. Full Pipeline Demonstration
    print("\n[5] Demonstrating Full Training Pipeline (End-to-End)...")
    # This function handles everything: data, model init, training loop, and submission generation
    # We use very few epochs and the debug subset to ensure it finishes quickly

    train_pipeline(
        debug_size=debug_size, epochs=1, batch_size=batch_size, load_cached_data=True
    )

    # Verify Submission
    submission_path = Config.SUBMISSION_PATH
    if os.path.exists(submission_path):
        df_sub = pd.read_csv(submission_path)
        print(f"    Submission file generated at: {submission_path}")
        print(f"    Submission shape: {df_sub.shape}")
        print(f"    First 5 rows:\n{df_sub.head()}")

        # Assertions
        assert Config.ID_COL in df_sub.columns, f"Missing ID column {Config.ID_COL}"
        assert (
            Config.TARGET_COL in df_sub.columns
        ), f"Missing Target column {Config.TARGET_COL}"
        assert (
            len(df_sub) == debug_size
        ), f"Expected {debug_size} predictions, found {len(df_sub)}"
        print("    Pipeline verification passed.")
    else:
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    print("\n=== All Demonstrations Completed Successfully ===")


if __name__ == "__main__":
    main()
