import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import provided library components
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import get_dataloader, MGMTDataset
from library.model import SDWIVNet
from library.engine import train_one_epoch, validate


def run_demonstration():
    # ==========================================
    # 1. Setup and Configuration Override
    # ==========================================
    print(">>> Setting up configuration for demonstration...")

    # Override Config for speed and demonstration purposes
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 12  # Small subset for quick execution
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Use main process for simplicity in demo
    Config.WORKING_DIR = "./working/demo_run"

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Initialize Logger
    logger = get_logger("demo", log_file=os.path.join(Config.WORKING_DIR, "demo.log"))

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    logger.info(f"Device: {Config.DEVICE}")

    # ==========================================
    # 2. Data Loading Demonstration
    # ==========================================
    logger.info(">>> Initializing Data Loaders...")

    # Load Train Loader
    # load_cached_data=False forces the raw DICOM processing logic to run
    train_loader = get_dataloader(
        split="train",
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        load_cached_data=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Load Validation Loader
    val_loader = get_dataloader(
        split="val",
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        load_cached_data=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Verify Data Shapes
    logger.info("Verifying training batch structure...")
    sample_imgs, sample_lbls = next(iter(train_loader))

    # Expected: (Batch, 9, 224, 224)
    # 9 channels = 3 modalities (FLAIR, T1wCE, T2w) * 3 depths (0.4, 0.5, 0.6)
    expected_shape = (Config.BATCH_SIZE, 9, Config.IMG_SIZE, Config.IMG_SIZE)

    if sample_imgs.shape != expected_shape:
        raise AssertionError(
            f"Expected image shape {expected_shape}, got {sample_imgs.shape}"
        )

    if sample_lbls.shape[0] != Config.BATCH_SIZE:
        raise AssertionError(
            f"Expected {Config.BATCH_SIZE} labels, got {sample_lbls.shape[0]}"
        )

    logger.info(f"Batch Image Shape: {sample_imgs.shape} (Verified)")
    logger.info(f"Batch Label Shape: {sample_lbls.shape} (Verified)")

    # ==========================================
    # 3. Model Instantiation
    # ==========================================
    logger.info(">>> Instantiating SD-WIVNet Model...")

    model = SDWIVNet()
    model = model.to(Config.DEVICE)

    # Verify Input Layer Modification
    # The first layer should accept 9 channels
    first_layer = model.backbone.conv_stem
    if first_layer.in_channels != 9:
        raise AssertionError(
            f"Model input channels expected 9, got {first_layer.in_channels}"
        )

    logger.info("Model input layer verified (9 channels).")

    # ==========================================
    # 4. Training Loop Demonstration
    # ==========================================
    logger.info(">>> Starting Training Loop (1 Epoch)...")

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    # Train for one epoch
    avg_train_loss = train_one_epoch(
        train_loader, model, criterion, optimizer, Config.DEVICE, epoch=1, logger=logger
    )

    logger.info(f"Training finished. Average Loss: {avg_train_loss:.4f}")

    # ==========================================
    # 5. Validation Demonstration
    # ==========================================
    logger.info(">>> Starting Validation...")

    val_loss, val_auc = validate(val_loader, model, criterion, Config.DEVICE, logger)

    logger.info(f"Validation finished. Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")

    # Basic sanity check on metrics
    if not (0 <= val_auc <= 1):
        raise AssertionError(f"AUC score {val_auc} is out of bounds [0, 1]")

    # ==========================================
    # 6. Inference & Submission Generation
    # ==========================================
    logger.info(">>> Generating Predictions for Test Set...")

    # Load Test Data
    test_loader = get_dataloader(
        split="test",
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        load_cached_data=False,
        num_workers=Config.NUM_WORKERS,
    )

    model.eval()
    predictions = []
    ids = []

    # Inference Loop
    with torch.no_grad():
        # We need to access IDs, so we iterate the dataset directly via loader if possible,
        # but the standard loader yields (images, targets).
        # We need to map predictions back to BraTS21ID.
        # The `load_data` function returns (ids, images, labels).
        # We can access the IDs from the cached file or by reloading the raw arrays.

        # Let's use the loader and assume sequential order matches the metadata/cache
        # (which is guaranteed by shuffle=False and deterministic loading).

        # Load IDs separately to map them
        from library.data import load_data

        test_ids, _, _ = load_data("test", load_cached_data=True)

        batch_idx = 0
        for images, _ in test_loader:
            images = images.to(Config.DEVICE)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            # Map current batch to IDs
            start_idx = batch_idx * Config.BATCH_SIZE
            end_idx = start_idx + len(probs)
            batch_ids = test_ids[start_idx:end_idx]

            for bid, prob in zip(batch_ids, probs):
                ids.append(bid)
                predictions.append(prob)

            batch_idx += 1

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": predictions})

    # Save Submission
    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    df_sub.to_csv(submission_path, index=False)

    logger.info(f"Submission saved to {submission_path}")
    logger.info("Head of submission:")
    logger.info(f"\n{df_sub.head()}")

    # Final Validation of Output
    if df_sub.isnull().values.any():
        raise AssertionError("Submission contains NaN values.")

    logger.info("Demonstration completed successfully.")


if __name__ == "__main__":
    run_demonstration()
