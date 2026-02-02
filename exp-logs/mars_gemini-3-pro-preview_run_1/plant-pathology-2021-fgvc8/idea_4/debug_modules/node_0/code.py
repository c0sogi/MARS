import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

# Import library modules
from library.config import Config
from library.utils import (
    seed_everything,
    calculate_f1_score,
    AverageMeter,
    get_logger,
    save_checkpoint,
    load_checkpoint,
)
from library.dataset import load_data, AppleDataset, get_transforms
from library.model import AppleConvNeXt
from library.engine import train_one_epoch, valid_one_epoch, inference

if __name__ == "__main__":
    print("Starting Library Usage Demonstration...")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup for Demo
    # -------------------------------------------------------------------------
    # Override Config defaults to run a fast demonstration
    Config.DEBUG = True
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8  # Small batch size for demo
    Config.WORKING_DIR = "./working/demo_execution"
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.LOG_PATH = os.path.join(Config.WORKING_DIR, "demo_log.txt")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Create necessary directories
    Config.create_dirs()

    # Initialize Logger
    logger = get_logger(Config.LOG_PATH)
    logger.info("Configuration configured for fast demonstration (DEBUG=True).")

    # -------------------------------------------------------------------------
    # 2. Utility Function Verification
    # -------------------------------------------------------------------------
    logger.info("\n--- Verifying Utilities ---")

    # Test seed_everything
    seed_everything(Config.SEED)

    # Test calculate_f1_score
    # Case: Perfect prediction
    y_true = np.array([[1, 0, 1], [0, 1, 0]])
    y_pred = np.array([[0.9, 0.1, 0.8], [0.2, 0.7, 0.1]])  # Probabilities
    score = calculate_f1_score(y_true, y_pred, threshold=0.5, average="macro")
    assert score == 1.0, f"Expected F1 score 1.0, got {score}"

    # Case: Zero prediction
    y_pred_bad = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
    score_bad = calculate_f1_score(y_true, y_pred_bad, threshold=0.5, average="macro")
    assert score_bad == 0.0, f"Expected F1 score 0.0, got {score_bad}"

    # Test AverageMeter
    meter = AverageMeter()
    meter.update(10, n=2)
    meter.update(20, n=2)
    assert meter.avg == 15.0, f"Expected AverageMeter avg 15.0, got {meter.avg}"

    logger.info("Utilities verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Data Loading & Dataset Verification
    # -------------------------------------------------------------------------
    logger.info("\n--- Verifying Data Loading & Dataset ---")

    # Load Training Data (Debug mode loads a small sample)
    df_train = load_data(Config.TRAIN_CSV, "train", debug=Config.DEBUG)
    assert not df_train.empty, "Training dataframe is empty."
    assert "file_path" in df_train.columns, "Missing 'file_path' column."
    assert "labels" in df_train.columns, "Missing 'labels' column."

    logger.info(f"Loaded {len(df_train)} training samples (Debug Mode).")

    # Initialize Dataset
    train_transforms = get_transforms("train")
    train_dataset = AppleDataset(df_train, transforms=train_transforms)

    # Verify Dataset Item
    image, target = train_dataset[0]

    # Check Image Tensor
    assert isinstance(
        image, torch.Tensor
    ), "Dataset should return a torch.Tensor for image."
    assert image.shape == (
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Expected image shape (3, {Config.IMAGE_SIZE}, {Config.IMAGE_SIZE}), got {image.shape}"

    # Check Target Tensor
    assert isinstance(
        target, torch.Tensor
    ), "Dataset should return a torch.Tensor for target."
    assert target.shape == (
        Config.NUM_CLASSES,
    ), f"Expected target shape ({Config.NUM_CLASSES},), got {target.shape}"

    # Check DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,  # Reduced workers for demo
        pin_memory=True,
    )

    batch_imgs, batch_targets = next(iter(train_loader))
    assert batch_imgs.shape[0] == Config.BATCH_SIZE or batch_imgs.shape[0] == len(
        df_train
    ), "Batch size mismatch in DataLoader."

    logger.info("Dataset and DataLoader verified successfully.")

    # -------------------------------------------------------------------------
    # 4. Model Architecture Verification
    # -------------------------------------------------------------------------
    logger.info("\n--- Verifying Model Architecture ---")

    model = AppleConvNeXt(pretrained=False)  # False for speed in demo
    model.to(Config.DEVICE)

    # Verify Forward Pass
    dummy_input = torch.randn(2, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE).to(
        Config.DEVICE
    )

    with torch.no_grad():
        outputs = model(dummy_input)

    assert outputs.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Expected model output shape (2, {Config.NUM_CLASSES}), got {outputs.shape}"

    logger.info("Model architecture and forward pass verified successfully.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Verification
    # -------------------------------------------------------------------------
    logger.info("\n--- Verifying Training Loop ---")

    # Setup components
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=Config.USE_AMP)

    # Run one epoch of training
    # Since DEBUG=True, the dataset is small (~100 samples), so this runs quickly
    train_loss = train_one_epoch(
        epoch=0,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        dataloader=train_loader,
        device=Config.DEVICE,
        scaler=scaler,
        logger=logger,
    )

    assert train_loss > 0, "Training loss should be positive."
    assert not np.isnan(train_loss), "Training loss resulted in NaN."

    logger.info(f"Training step completed. Loss: {train_loss:.4f}")

    # -------------------------------------------------------------------------
    # 6. Validation Loop Verification
    # -------------------------------------------------------------------------
    logger.info("\n--- Verifying Validation Loop ---")

    # Load Validation Data
    df_val = load_data(Config.VAL_CSV, "val", debug=Config.DEBUG)
    val_dataset = AppleDataset(df_val, transforms=get_transforms("valid"))
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    # Run validation
    val_loss, val_f1 = valid_one_epoch(
        epoch=0,
        model=model,
        criterion=criterion,
        dataloader=val_loader,
        device=Config.DEVICE,
        logger=logger,
    )

    assert val_loss > 0, "Validation loss should be positive."
    assert 0.0 <= val_f1 <= 1.0, "F1 score must be between 0 and 1."

    logger.info(f"Validation step completed. Loss: {val_loss:.4f}, F1: {val_f1:.4f}")

    # -------------------------------------------------------------------------
    # 7. Checkpoint & Inference Verification
    # -------------------------------------------------------------------------
    logger.info("\n--- Verifying Checkpoint & Inference ---")

    # Save Checkpoint
    save_checkpoint(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_f1": val_f1,
        },
        Config.MODEL_PATH,
    )
    assert os.path.exists(Config.MODEL_PATH), "Checkpoint file was not created."

    # Load Checkpoint
    loaded_checkpoint = load_checkpoint(model, Config.MODEL_PATH, Config.DEVICE)
    assert "model_state_dict" in loaded_checkpoint, "Invalid checkpoint format."

    # Inference on Test Data (using val data as proxy for demo)
    # We use output_label=False to simulate test dataset behavior
    test_dataset = AppleDataset(
        df_val, transforms=get_transforms("test"), output_label=False
    )
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    preds = inference(model, test_loader, Config.DEVICE)

    assert preds.shape == (
        len(df_val),
        Config.NUM_CLASSES,
    ), f"Prediction shape mismatch. Expected ({len(df_val)}, {Config.NUM_CLASSES}), got {preds.shape}"
    assert (
        preds.min() >= 0.0 and preds.max() <= 1.0
    ), "Predictions should be probabilities [0, 1]."

    logger.info("Checkpoint saving/loading and inference verified successfully.")

    print("\nAll library components demonstrated and verified successfully.")
