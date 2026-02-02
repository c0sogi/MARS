import os
import shutil
import torch
import numpy as np
import pandas as pd
import sys

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, AverageMeter, calculate_accuracy
from library.features import FeatureEngineer
from library.data_loader import get_dataloaders
from library.model import ParallelDCNResNeXt, generate_submission
from library.trainer import Trainer


def main():
    print("Starting demonstration of library components...")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("\n[1] Configuring environment for demo...")

    # Override Config for speed and isolation
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 500  # Small subset for quick execution
    Config.EPOCHS = 1  # Single epoch to prove training loop works
    Config.BATCH_SIZE = 32  # Small batch size for debug
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure clean slate
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed
    seed_everything(Config.SEED)
    print("Configuration updated: DEBUG=True, EPOCHS=1")

    # ==========================================
    # 2. Verify Utilities
    # ==========================================
    print("\n[2] Verifying utils.py...")

    # Test AverageMeter
    meter = AverageMeter()
    meter.update(10, n=1)
    meter.update(20, n=1)
    assert meter.avg == 15.0, f"AverageMeter failed: expected 15.0, got {meter.avg}"
    print("AverageMeter verified.")

    # Test calculate_accuracy
    logits = torch.tensor(
        [[2.0, 0.5, 0.1], [0.1, 2.0, 0.5]]
    )  # Classes 0 and 1 predicted
    targets = torch.tensor([0, 1])
    acc = calculate_accuracy(logits, targets)
    assert acc == 1.0, f"calculate_accuracy failed: expected 1.0, got {acc}"
    print("calculate_accuracy verified.")

    # ==========================================
    # 3. Data Loading & Feature Engineering
    # ==========================================
    print("\n[3] Verifying data_loader.py and features.py...")

    # This will trigger FeatureEngineer.process_data internally
    # load_cached_data=False forces the pipeline to run from scratch
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=0,  # Use 0 workers for simple debugging to avoid multiprocessing overhead
        load_cached_data=False,
    )

    # Verify Train Loader
    X_batch, y_batch = next(iter(train_loader))
    print(f"Train Batch Shapes - X: {X_batch.shape}, y: {y_batch.shape}")

    assert X_batch.dim() == 2, "Input features must be 2D (Batch, Features)"
    assert y_batch.dim() == 1, "Targets must be 1D (Batch)"
    assert X_batch.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"

    # Calculate input dimension for model
    input_dim = X_batch.shape[1]
    print(f"Detected Input Dimension: {input_dim}")

    # Verify Test Loader (should return X and ids)
    X_test_batch, ids_batch = next(iter(test_loader))
    assert ids_batch.dim() == 1, "Test IDs must be 1D"
    print("DataLoaders verified.")

    # ==========================================
    # 4. Model Architecture
    # ==========================================
    print("\n[4] Verifying model.py (ParallelDCNResNeXt)...")

    model = ParallelDCNResNeXt(input_dim=input_dim, num_classes=Config.NUM_CLASSES)
    model.to(Config.DEVICE)

    # Forward pass check
    dummy_input = X_batch.to(Config.DEVICE)
    output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, Config.NUM_CLASSES)}, got {output.shape}"

    # Check for NaN
    assert not torch.isnan(output).any(), "Model produced NaN outputs"
    print("Model forward pass verified.")

    # ==========================================
    # 5. Training Loop
    # ==========================================
    print("\n[5] Verifying trainer.py...")

    trainer = Trainer(model, train_loader, val_loader)

    # Run training (1 epoch as configured)
    trained_model = trainer.fit()

    assert trained_model is not None, "Trainer.fit() returned None"
    print("Training loop completed successfully.")

    # ==========================================
    # 6. Inference & Submission
    # ==========================================
    print("\n[6] Verifying submission generation...")

    generate_submission(
        trained_model,
        test_loader,
        device=Config.DEVICE,
        output_path=Config.SUBMISSION_PATH,
    )

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission File Shape: {df_sub.shape}")
    print(f"Columns: {df_sub.columns.tolist()}")

    assert (
        "Id" in df_sub.columns and "Cover_Type" in df_sub.columns
    ), "Missing required columns in submission"
    assert len(df_sub) > 0, "Submission file is empty"

    # Verify IDs match the debug sample size (or close to it depending on batching/dropping)
    # Note: Test loader does not drop last, but debug mode slices data.
    # Config.DEBUG_SAMPLE_SIZE was set to 500.
    assert (
        len(df_sub) == Config.DEBUG_SAMPLE_SIZE
    ), f"Submission rows ({len(df_sub)}) do not match debug sample size ({Config.DEBUG_SAMPLE_SIZE})"

    print("Submission generation verified.")

    # ==========================================
    # 7. Cleanup
    # ==========================================
    print("\n[7] Cleaning up...")
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    print("Cleanup complete.")

    print("\nAll demonstrations passed successfully.")


if __name__ == "__main__":
    main()
