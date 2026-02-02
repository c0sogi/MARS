import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.utils import (
    AverageMeter,
    calculate_f1_score,
    save_checkpoint,
    get_logger,
    seed_everything,
)
from library.dataset import AppleDataset, get_transforms, load_data
from library.model import AppleConvNeXt


def train_one_epoch(
    epoch, model, optimizer, scheduler, criterion, dataloader, device, scaler, logger
):
    """
    Handles the training of a single epoch.
    """
    model.train()
    losses = AverageMeter()
    start = time.time()

    for step, (images, targets) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = images.size(0)

        # Apply Label Smoothing manually for BCEWithLogitsLoss
        # Formula: new_targets = targets * (1 - epsilon) + 0.5 * epsilon
        if Config.LABEL_SMOOTHING > 0:
            targets = (
                targets * (1 - Config.LABEL_SMOOTHING) + 0.5 * Config.LABEL_SMOOTHING
            )

        # Mixed Precision Training
        with autocast(enabled=Config.USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, targets)

        losses.update(loss.item(), batch_size)

        # Backpropagation with Scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    # Log metrics
    elapsed = time.time() - start
    logger.info(
        f"Epoch {epoch+1} | Train Loss: {losses.avg:.6f} | Time: {elapsed:.1f}s"
    )

    return losses.avg


def valid_one_epoch(epoch, model, criterion, dataloader, device, logger):
    """
    Handles the validation of a single epoch.
    """
    model.eval()
    losses = AverageMeter()
    preds_all = []
    targets_all = []
    start = time.time()

    with torch.no_grad():
        for step, (images, targets) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            batch_size = images.size(0)

            with autocast(enabled=Config.USE_AMP):
                outputs = model(images)
                loss = criterion(outputs, targets)

            losses.update(loss.item(), batch_size)

            # Apply sigmoid to convert logits to probabilities
            preds_all.append(torch.sigmoid(outputs).cpu().numpy())
            targets_all.append(targets.cpu().numpy())

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)

    # Calculate F1 Score
    val_f1 = calculate_f1_score(targets_all, preds_all, threshold=Config.THRESHOLD)

    elapsed = time.time() - start
    logger.info(
        f"Epoch {epoch+1} | Valid Loss: {losses.avg:.6f} | Valid F1: {val_f1:.6f} | Time: {elapsed:.1f}s"
    )

    return losses.avg, val_f1


def inference(model, dataloader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    preds_all = []

    with torch.no_grad():
        for step, images in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            with autocast(enabled=Config.USE_AMP):
                outputs = model(images)
            preds_all.append(torch.sigmoid(outputs).cpu().numpy())

    return np.concatenate(preds_all)


def run_training():
    """
    Main driver function to run the training pipeline, handle early stopping,
    and generate the submission file.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    logger = get_logger(Config.LOG_PATH)
    Config.create_dirs()

    logger.info(f"Starting training with model: {Config.MODEL_NAME}")
    logger.info(f"Device: {Config.DEVICE}")

    # 2. Data Loading
    logger.info("Loading datasets...")
    df_train = load_data(Config.TRAIN_CSV, "train", debug=Config.DEBUG)
    df_val = load_data(Config.VAL_CSV, "val", debug=Config.DEBUG)

    train_dataset = AppleDataset(df_train, transforms=get_transforms("train"))
    val_dataset = AppleDataset(df_val, transforms=get_transforms("valid"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # 3. Model & Optimization
    model = AppleConvNeXt(pretrained=Config.PRETRAINED)
    model.to(Config.DEVICE)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.SCHEDULER_MIN_LR
    )

    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=Config.USE_AMP)

    # 4. Training Loop
    best_f1 = 0.0
    patience = 5  # Early stopping patience
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        logger.info(f"\n--- Epoch {epoch+1}/{Config.EPOCHS} ---")

        # Train
        train_loss = train_one_epoch(
            epoch,
            model,
            optimizer,
            scheduler,
            criterion,
            train_loader,
            Config.DEVICE,
            scaler,
            logger,
        )

        # Validate
        val_loss, val_f1 = valid_one_epoch(
            epoch, model, criterion, val_loader, Config.DEVICE, logger
        )

        # Scheduler Step
        scheduler.step()

        # Save Best Model & Early Stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            logger.info(f"New Best F1: {best_f1:.6f}. Saving model...")
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_f1": best_f1,
                },
                Config.MODEL_PATH,
            )
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            logger.info("Early stopping triggered.")
            break

    logger.info(f"Training Complete. Best F1: {best_f1:.6f}")

    # 5. Inference & Submission
    logger.info("Starting Inference on Test Set...")

    # Load Best Model
    checkpoint = torch.load(Config.MODEL_PATH, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load Test Data
    df_test = load_data(Config.TEST_CSV, "test", debug=Config.DEBUG)
    test_dataset = AppleDataset(
        df_test, transforms=get_transforms("test"), output_label=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Generate Predictions
    preds = inference(model, test_loader, Config.DEVICE)

    # Format Submission
    logger.info("Generating Submission File...")
    submission_rows = []

    for i, row in df_test.iterrows():
        img_id = row["image"]
        probs = preds[i]

        # Get indices where prob > threshold
        active_indices = np.where(probs > Config.THRESHOLD)[0]

        if len(active_indices) == 0:
            # Fallback: if no class exceeds threshold, take the argmax to avoid empty prediction
            top_idx = np.argmax(probs)
            labels_str = Config.CLASSES[top_idx]
        else:
            labels = [Config.CLASSES[idx] for idx in active_indices]
            labels_str = " ".join(labels)

        submission_rows.append({"image": img_id, "labels": labels_str})

    df_sub = pd.DataFrame(submission_rows)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
