import sys
import os
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel

# Ensure library is in path
sys.path.append(".")

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, setup_logger
from library.data import get_dataloaders
from library.model import IcebergResNet
from library.sam import SAM
from library.engine import (
    train_one_epoch,
    validate_tta,
    predict_tta,
    update_swa_bn,
    save_submission,
)


def main():
    # =========================================================================
    # 1. Setup and Configuration
    # =========================================================================
    # Override specific configs for the demo execution to ensure speed
    Config.WORK_DIR = "./working/demo_execution"
    Config.BATCH_SIZE = 32  # Reduced batch size for safety

    # Create directories
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    # Initialize Logger
    logger = setup_logger("demo", os.path.join(Config.WORK_DIR, "demo.log"))
    logger.info("Starting demo execution...")

    # Set Seeds
    seed_everything(Config.SEED)
    logger.info(f"Random seed set to {Config.SEED}")

    # Device selection
    device = Config.DEVICE
    logger.info(f"Using device: {device}")

    # =========================================================================
    # 2. Data Loading
    # =========================================================================
    logger.info("Initializing DataLoaders...")

    # Get dataloaders (using cached data if available to speed up)
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=True,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # Verification: Check if loaders are populated
    if len(train_loader) == 0:
        raise ValueError("Train loader is empty.")
    if len(val_loader) == 0:
        raise ValueError("Validation loader is empty.")
    if len(test_loader) == 0:
        raise ValueError("Test loader is empty.")

    # Verification: Check batch structure
    # Fetch one batch to verify shapes
    sample_imgs, sample_angs, sample_lbls = next(iter(train_loader))

    # Expected: (Batch, 3, 224, 224)
    assert sample_imgs.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect image shape: {sample_imgs.shape}"
    # Expected: (Batch,)
    assert sample_angs.shape == (
        Config.BATCH_SIZE,
    ), f"Incorrect angle shape: {sample_angs.shape}"

    logger.info("DataLoaders initialized and verified.")

    # =========================================================================
    # 3. Model Initialization
    # =========================================================================
    logger.info("Initializing IcebergResNet model...")
    model = IcebergResNet()
    model.to(device)

    # Verification: Dummy Forward Pass
    with torch.no_grad():
        # Move sample data to device
        dummy_imgs = sample_imgs.to(device)
        dummy_angs = sample_angs.to(device)

        # Forward
        dummy_out = model(dummy_imgs, dummy_angs)

        # Check output shape: (Batch, 1)
        assert dummy_out.shape == (
            Config.BATCH_SIZE,
            1,
        ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, 1)}, got {dummy_out.shape}"

    logger.info("Model initialized and architecture verified.")

    # =========================================================================
    # 4. Optimizer Setup (SAM)
    # =========================================================================
    logger.info("Setting up SAM optimizer...")
    base_optimizer = AdamW
    optimizer = SAM(
        model.parameters(),
        base_optimizer,
        lr=Config.LR,
        rho=Config.SAM_RHO,
        weight_decay=Config.WEIGHT_DECAY,
    )

    # =========================================================================
    # 5. Training Loop (Demo)
    # =========================================================================
    logger.info("Starting training demo (2 epochs)...")

    # Train for 2 epochs to demonstrate functionality
    for epoch in range(1, 3):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        logger.info(f"Epoch {epoch} Training Loss: {train_loss:.6f}")

        # Verification: Loss should be a valid number
        assert not np.isnan(train_loss), f"Training loss is NaN at epoch {epoch}"

    # =========================================================================
    # 6. Validation (TTA)
    # =========================================================================
    logger.info("Running validation with TTA...")
    val_loss = validate_tta(model, val_loader, device)
    logger.info(f"Validation Log Loss: {val_loss:.6f}")
    assert not np.isnan(val_loss), "Validation loss is NaN"

    # =========================================================================
    # 7. SWA Integration Demo
    # =========================================================================
    logger.info("Demonstrating SWA setup and BN update...")

    # Create SWA Model wrapper
    swa_model = AveragedModel(model)

    # Update parameters (In a real scenario, this happens over multiple epochs)
    # Here we just copy the current model state
    swa_model.update_parameters(model)

    # Update BatchNorm statistics
    # This uses the custom function from library.engine to handle the (img, angle) input
    update_swa_bn(swa_model, train_loader, device)

    logger.info("SWA BN statistics updated.")

    # =========================================================================
    # 8. Inference (Prediction)
    # =========================================================================
    logger.info("Generating predictions on test set...")

    # Predict using the SWA model (or standard model)
    preds = predict_tta(swa_model, test_loader, device)

    # Verification: Prediction count and range
    assert len(preds) == len(
        test_ids
    ), f"Prediction count ({len(preds)}) does not match Test ID count ({len(test_ids)})"

    assert np.all(
        (preds >= 0.0) & (preds <= 1.0)
    ), "Predictions contain values outside [0, 1] range."

    logger.info(f"Generated {len(preds)} predictions.")

    # =========================================================================
    # 9. Submission
    # =========================================================================
    submission_path = os.path.join(Config.WORK_DIR, "submission", "submission.csv")
    logger.info(f"Saving submission to {submission_path}...")

    save_submission(preds, test_ids, submission_path)

    # Verification: Check file existence and format
    if not os.path.exists(submission_path):
        raise FileNotFoundError("Submission file was not created.")

    df_sub = pd.read_csv(submission_path)
    if list(df_sub.columns) != ["id", "is_iceberg"]:
        raise ValueError(f"Incorrect submission columns: {df_sub.columns}")
    if len(df_sub) != len(test_ids):
        raise ValueError("Submission row count mismatch.")

    logger.info("Demo execution completed successfully.")


if __name__ == "__main__":
    main()
