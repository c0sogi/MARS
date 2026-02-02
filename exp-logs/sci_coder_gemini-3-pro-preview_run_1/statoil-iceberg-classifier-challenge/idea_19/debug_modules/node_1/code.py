import os
import torch
import pandas as pd
import numpy as np
import logging
import shutil

# Import from the provided library files
from library.utils import set_seed, setup_logger
from library.data import get_dataloaders
from library.model import IcebergResNet18GeM
from library.training import Trainer
from library.inference import predict_tta, select_pseudo_labels

# Configuration for the demo
WORKING_DIR = "./working/demo_execution"
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")
LOG_FILE = os.path.join(WORKING_DIR, "demo.log")
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# Hyperparameters for the demo (optimized for speed)
BATCH_SIZE = 32
NUM_WORKERS = 2
LEARNING_RATE = 1e-3
EPOCHS = 2  # Kept low for demonstration speed
SWA_EPOCHS = 1  # Kept low for demonstration speed


def main():
    # 1. Setup Environment
    # ====================
    print("Step 1: Setting up environment...")

    # Clean up previous run if exists
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(42)

    # Setup Logger
    logger = setup_logger("DemoLogger", LOG_FILE)
    logger.info("Environment setup complete.")

    # Determine Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 2. Data Loading
    # ===============
    print("Step 2: Loading data...")
    # get_dataloaders handles caching, splitting based on metadata, and augmentation
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        load_cached_data=True,  # Will cache to ./working/idea_19/ by default in data.py
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")

    # Verification: Check batch structure
    sample_batch = next(iter(train_loader))
    # Expecting: images, angles, labels, ids
    assert (
        len(sample_batch) == 4
    ), "Train loader should return (images, angles, labels, ids)"
    images, angles, labels, ids = sample_batch
    assert images.shape == (
        BATCH_SIZE,
        3,
        224,
        224,
    ), f"Unexpected image shape: {images.shape}"
    assert angles.shape == (BATCH_SIZE,), f"Unexpected angle shape: {angles.shape}"

    # 3. Model Initialization
    # =======================
    print("Step 3: Initializing model...")
    # Initialize the custom ResNet18 with GeM pooling
    model = IcebergResNet18GeM(pretrained=True)
    model = model.to(device)
    logger.info("Model initialized successfully.")

    # 4. Training Loop (Standard)
    # ===========================
    print("Step 4: Starting standard training...")
    trainer = Trainer(model, device, logger, learning_rate=LEARNING_RATE)

    # Run standard training
    best_loss = trainer.fit(
        train_loader, val_loader, epochs=EPOCHS, checkpoint_dir=CHECKPOINT_DIR
    )

    logger.info(f"Standard training complete. Best Val Loss: {best_loss:.4f}")

    # Verification: Check if checkpoints are created
    assert os.path.exists(
        os.path.join(CHECKPOINT_DIR, "best_model.pth")
    ), "best_model.pth not found"
    assert os.path.exists(
        os.path.join(CHECKPOINT_DIR, "checkpoint.pth")
    ), "checkpoint.pth not found"

    # 5. Training Loop (SWA)
    # ======================
    print("Step 5: Starting SWA training...")
    # Run Stochastic Weight Averaging phase
    trainer.fit_swa(
        train_loader, val_loader, swa_epochs=SWA_EPOCHS, checkpoint_dir=CHECKPOINT_DIR
    )

    # Verification: Check if SWA model is saved
    assert os.path.exists(
        os.path.join(CHECKPOINT_DIR, "swa_model.pth")
    ), "swa_model.pth not found"

    # 6. Inference & Submission
    # =========================
    print("Step 6: Generating submission...")
    # Generate submission file using the trainer's utility
    trainer.generate_submission(test_loader, SUBMISSION_FILE)

    # Verification: Check submission file
    assert os.path.exists(SUBMISSION_FILE), "Submission file was not created."
    df_sub = pd.read_csv(SUBMISSION_FILE)
    assert list(df_sub.columns) == [
        "id",
        "is_iceberg",
    ], "Submission columns are incorrect."
    assert len(df_sub) > 0, "Submission file is empty."
    logger.info(f"Submission generated with {len(df_sub)} rows.")

    # 7. Advanced Inference Utilities
    # ===============================
    print("Step 7: Demonstrating advanced inference utilities...")

    # Demonstrate direct use of predict_tta from library.inference
    # We'll use the validation loader for this demonstration to verify logic
    val_preds = predict_tta(model, val_loader, device)
    assert len(val_preds) > 0, "predict_tta returned empty results."

    # Demonstrate Pseudo-Label Selection
    # To simulate an ensemble, we'll create a list containing the same predictions twice
    # In a real scenario, these would be predictions from different models
    ensemble_preds = [val_preds, val_preds]

    # Select pseudo labels (using loose thresholds for demo purposes to ensure some are selected)
    pseudo_df = select_pseudo_labels(
        ensemble_preds, confidence_threshold=0.8, variance_threshold=0.05
    )

    logger.info(f"Pseudo-labels selected: {len(pseudo_df)}")

    # Verification: Check pseudo label dataframe structure
    if len(pseudo_df) > 0:
        assert "id" in pseudo_df.columns and "is_iceberg" in pseudo_df.columns
        # Check values are binary (0.0 or 1.0)
        unique_vals = pseudo_df["is_iceberg"].unique()
        for v in unique_vals:
            assert v in [
                0.0,
                1.0,
            ], f"Pseudo labels must be hard labels (0 or 1), found {v}"

    print("\nAll demonstrations completed successfully.")
    print(f"Artifacts stored in: {WORKING_DIR}")


if __name__ == "__main__":
    main()
