import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, load_checkpoint
from library.model_layers import DualPooling, WideBlock
from library.model_arch import DMWBN
from library.data_loader import process_data, get_loaders
from library.engine import Trainer


def run_demo():
    # ==========================================
    # 1. SETUP & CONFIGURATION OVERRIDES
    # ==========================================
    print("\n[1] Setting up configuration and environment...")

    # Set deterministic behavior
    seed_everything(42)

    # Override Config paths to use a separate demo directory in ./working
    # This ensures we don't overwrite existing artifacts from other runs
    DEMO_DIR = "./working/demo_execution"
    Config.WORK_DIR = DEMO_DIR
    Config.CACHE_FILE = os.path.join(DEMO_DIR, "cache", "processed_data.npz")
    Config.SUBMISSION_DIR = DEMO_DIR
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Reduce epochs for speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 16  # Smaller batch size for quick iteration

    # Ensure directories exist
    os.makedirs(os.path.dirname(Config.CACHE_FILE), exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")
    print(f"    Demo Directory: {DEMO_DIR}")

    # ==========================================
    # 2. VERIFY CUSTOM LAYERS (LOGIC CHECK)
    # ==========================================
    print("\n[2] Verifying custom model layers...")

    # Test DualPooling
    # Logic: Concatenates MaxPool and -MaxPool(-x), so channels should double, spatial dim halve
    dummy_input = torch.randn(2, 64, 75, 75)
    pool_layer = DualPooling(kernel_size=2, stride=2)
    pool_out = pool_layer(dummy_input)

    print(f"    DualPooling Input: {dummy_input.shape}")
    print(f"    DualPooling Output: {pool_out.shape}")

    assert pool_out.shape[1] == 64 * 2, "DualPooling failed to double channels"
    assert pool_out.shape[2] == 37, "DualPooling spatial height incorrect (floor(75/2))"

    # Test WideBlock
    # Logic: Conv -> BN -> ReLU -> CBAM -> DualPooling
    # Input C -> Output 2*C_out due to pooling
    wide_block = WideBlock(in_channels=64, out_channels=32)
    block_out = wide_block(dummy_input)

    print(f"    WideBlock Input: {dummy_input.shape}")
    print(f"    WideBlock Output: {block_out.shape}")

    assert block_out.shape[1] == 32 * 2, "WideBlock output channels incorrect"
    assert block_out.shape[2] == 37, "WideBlock spatial dimension incorrect"

    print("    Layer verification passed.")

    # ==========================================
    # 3. DATA PIPELINE DEMONSTRATION
    # ==========================================
    print("\n[3] Executing Data Pipeline...")

    # Force reprocessing to demonstrate the logic (load_cached_data=False)
    # In a real scenario, we would use True to save time
    print("    Processing data from JSON inputs...")
    data_artifacts = process_data(load_cached_data=False)

    assert "X_train" in data_artifacts
    assert "X_test" in data_artifacts

    # Get DataLoaders
    print("    Creating DataLoaders...")
    train_loader, val_loader, test_loader = get_loaders(
        fold_idx=0, load_cached_data=True
    )

    # Verify Batch Structure
    images, angles, labels = next(iter(train_loader))
    print(
        f"    Batch Shapes -> Images: {images.shape}, Angles: {angles.shape}, Labels: {labels.shape}"
    )

    # Assertions for data integrity
    assert images.shape[1] == 3, "Images should have 3 channels (HH, HV, Avg)"
    assert images.shape[2] == Config.IMG_HEIGHT, "Incorrect image height"
    assert not torch.isnan(angles).any(), "Batch contains NaN incidence angles"

    print("    Data pipeline verification passed.")

    # ==========================================
    # 4. MODEL TRAINING DEMONSTRATION
    # ==========================================
    print("\n[4] Initializing Model and Training Loop...")

    model = DMWBN().to(device)

    # Optimizer and Scheduler setup
    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    # Trainer Initialization
    trainer = Trainer(
        model=model, device=device, optimizer=optimizer, scheduler=scheduler
    )

    # Run Training
    save_path = os.path.join(DEMO_DIR, "model_fold_0.pth")
    print(f"    Starting training for {Config.EPOCHS} epochs...")
    best_loss = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=Config.EPOCHS,
        patience=Config.EARLY_STOPPING_PATIENCE,
        save_path=save_path,
    )

    print(f"    Training complete. Best Validation Loss: {best_loss:.4f}")
    assert os.path.exists(save_path), "Model checkpoint was not saved."

    # ==========================================
    # 5. INFERENCE & SUBMISSION GENERATION
    # ==========================================
    print("\n[5] Running Inference and Generating Submission...")

    # Load the best model
    model = DMWBN().to(device)
    model = load_checkpoint(model, save_path, device)
    model.eval()

    predictions = []
    ids = []

    # Inference Loop
    with torch.no_grad():
        for batch_idx, (images, angles, _) in enumerate(test_loader):
            images = images.to(device)
            angles = angles.to(device)

            # Forward pass
            outputs = model(images, angles)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            # Get IDs for this batch
            # Note: The DataLoader shuffles=False for test, but we need to map IDs correctly.
            # The IcebergDataset for test doesn't return IDs in __getitem__,
            # so we rely on the order matching the source arrays.
            # We can retrieve the IDs from the cached data artifacts for verification.
            start_idx = batch_idx * Config.BATCH_SIZE
            end_idx = start_idx + len(images)
            batch_ids = data_artifacts["ids_test"][start_idx:end_idx]

            predictions.extend(probs)
            ids.extend(batch_ids)

    # Create DataFrame
    df_sub = pd.DataFrame({"id": ids, "is_iceberg": predictions})

    # Format check
    print(f"    Submission Shape: {df_sub.shape}")
    print(f"    First 5 predictions:\n{df_sub.head()}")

    # Save submission
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"    Submission saved to: {Config.SUBMISSION_PATH}")

    # Verify against sample submission format
    sample_sub = pd.read_csv("./input/sample_submission.csv")
    assert (
        df_sub.shape == sample_sub.shape
    ), f"Submission shape mismatch. Expected {sample_sub.shape}, got {df_sub.shape}"
    assert list(df_sub.columns) == list(sample_sub.columns), "Column name mismatch"

    print("\n[SUCCESS] Demo completed successfully.")


if __name__ == "__main__":
    run_demo()
