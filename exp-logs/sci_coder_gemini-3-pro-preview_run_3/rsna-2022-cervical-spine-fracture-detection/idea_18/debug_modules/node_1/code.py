import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library components
from library.config import Config
from library.utils import seed_everything, setup_logger
from library.dataset import CervicalSpineDataset
from library.model import DynamicDepthConvNeXt
from library.loss import ImplicitWeightedLoss
from library.engine import fit
from library.inference import run_inference


def main():
    # --- 1. Configuration & Setup ---
    # Set random seed for reproducibility
    seed_everything(42)

    # Override Config for a fast demonstration
    # Disable pretrained weights to avoid download/network issues
    Config.PRETRAINED = False
    # Reduce image size and sequence length for faster processing
    Config.IMAGE_SIZE = (128, 128)
    Config.SEQ_LENGTH = 16
    # Reduce batch size and workers
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 2
    # Enable Debug Mode: Limits data to 4 samples and training to 1 epoch
    Config.set_debug_mode(debug=True, data_size=4, epochs=1)

    # Define paths
    log_path = os.path.join(Config.WORKING_DIR, "demo.log")
    model_save_path = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    submission_path = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Setup Logger
    logger = setup_logger(log_path)
    logger.info("Starting End-to-End Pipeline Demonstration")

    # --- 2. Data Loading ---
    logger.info("Initializing Training Dataset...")
    # Initialize dataset with explicit sequence length to match our config override
    train_dataset = CervicalSpineDataset(
        mode="train",
        load_cached_data=False,  # Force processing for demonstration
        seq_length=Config.SEQ_LENGTH,
    )

    # Validate Dataset Size (Debug mode should limit it)
    assert (
        len(train_dataset) == Config.DEBUG_DATA_SIZE
    ), f"Dataset size {len(train_dataset)} does not match debug size {Config.DEBUG_DATA_SIZE}"

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Fetch a single batch to verify data pipeline
    logger.info("Fetching a batch of data...")
    inputs, targets = next(iter(train_loader))

    # Validate Input/Target Shapes
    # Inputs: (Batch, Seq, Channels, H, W)
    expected_input_shape = (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        3,
        Config.IMAGE_SIZE[0],
        Config.IMAGE_SIZE[1],
    )
    # Targets: (Batch, 8)
    expected_target_shape = (Config.BATCH_SIZE, Config.NUM_CLASSES)

    if inputs.shape != expected_input_shape:
        raise AssertionError(
            f"Input shape mismatch. Expected {expected_input_shape}, got {inputs.shape}"
        )
    if targets.shape != expected_target_shape:
        raise AssertionError(
            f"Target shape mismatch. Expected {expected_target_shape}, got {targets.shape}"
        )

    logger.info(
        f"Data Batch Verified. Inputs: {inputs.shape}, Targets: {targets.shape}"
    )

    # --- 3. Model Initialization & Forward Pass ---
    logger.info("Initializing Model...")
    device = Config.DEVICE
    model = DynamicDepthConvNeXt(
        backbone_name=Config.BACKBONE,
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES,
    ).to(device)

    # Run Forward Pass
    logger.info("Running Forward Pass...")
    inputs = inputs.to(device)
    targets = targets.to(device)

    logits = model(inputs)

    # Validate Model Output
    if logits.shape != expected_target_shape:
        raise AssertionError(
            f"Model output shape mismatch. Expected {expected_target_shape}, got {logits.shape}"
        )
    if torch.isnan(logits).any():
        raise AssertionError("Model produced NaN logits.")

    logger.info("Forward Pass Successful.")

    # --- 4. Loss Calculation ---
    logger.info("Calculating Loss...")
    loss_fn = ImplicitWeightedLoss()
    loss = loss_fn(logits, targets)

    # Validate Loss
    if loss.dim() != 0:
        raise AssertionError("Loss must be a scalar.")
    if loss.item() < 0:
        raise AssertionError("Loss must be non-negative.")

    logger.info(f"Loss Calculation Successful. Value: {loss.item():.4f}")

    # --- 5. Training Loop ---
    logger.info("Starting Training Loop (1 Epoch)...")

    # Initialize Validation Loader
    val_dataset = CervicalSpineDataset(
        mode="val", load_cached_data=False, seq_length=Config.SEQ_LENGTH
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Run Fit
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,
        loss_fn=loss_fn,
        device=device,
        epochs=Config.EPOCHS,
        patience=1,
        save_path=model_save_path,
    )

    if not os.path.exists(model_save_path):
        raise FileNotFoundError("Model checkpoint was not saved.")

    logger.info("Training Loop Completed.")

    # --- 6. Inference ---
    logger.info("Running Inference...")

    # Run inference using the trained model
    # Note: This uses the test metadata (sliced by debug mode) and sample_submission.csv
    run_inference(
        model_path=model_save_path,
        output_path=submission_path,
        batch_size=Config.BATCH_SIZE,
        device=device,
    )

    if not os.path.exists(submission_path):
        raise FileNotFoundError("Submission file was not created.")

    # Validate Submission File Structure
    sub_df = pd.read_csv(submission_path)
    required_cols = ["row_id", "fractured"]
    if not all(col in sub_df.columns for col in required_cols):
        raise AssertionError(f"Submission missing columns. Found: {sub_df.columns}")

    # Check if we have rows (should contain all rows from sample_submission, even if we only predicted 4 studies)
    if len(sub_df) == 0:
        raise AssertionError("Submission file is empty.")

    logger.info(f"Inference Completed. Submission saved to {submission_path}")
    logger.info("Demo Script Finished Successfully.")


if __name__ == "__main__":
    main()
