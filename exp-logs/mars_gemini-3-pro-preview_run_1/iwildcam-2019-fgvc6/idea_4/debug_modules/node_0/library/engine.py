import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import get_logger, calculate_metrics
from library.loss import FocalLoss, get_class_weights
from library.model import AnimalModel, ModelEMA

logger = get_logger("engine")


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    ema_model: ModelEMA = None,
):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update EMA model
        if ema_model:
            ema_model.update(model)

        # Metrics tracking
        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader)
    epoch_f1 = calculate_metrics(all_targets, all_preds)

    logger.info(
        f"Epoch {epoch} Training - Loss: {epoch_loss:.6f}, Macro F1: {epoch_f1:.6f}"
    )

    return epoch_loss, epoch_f1


def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())

    val_loss = running_loss / len(loader)
    val_f1 = calculate_metrics(all_targets, all_preds)

    # Print full precision as requested
    logger.info(f"Validation - Loss: {val_loss}, Macro F1: {val_f1}")

    return val_loss, val_f1


def predict_and_submit(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_path: str = Config.SUBMISSION_PATH,
):
    """
    Generates predictions for the test set and saves them to a CSV file.
    """
    model.eval()
    ids = []
    predictions = []

    logger.info("Starting prediction on test set...")

    with torch.no_grad():
        for images, batch_ids in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            ids.extend(batch_ids)
            predictions.extend(preds)

    # Create submission DataFrame
    # Using 'Id' and 'Predicted' as per task description
    df = pd.DataFrame({"Id": ids, "Predicted": predictions})

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}")


def run_training(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_epochs: int = Config.EPOCHS,
    patience: int = 3,
):
    """
    Main orchestration function for training.
    """
    device = Config.DEVICE
    logger.info(f"Starting training on device: {device}")

    # 1. Calculate Class Weights for Focal Loss
    # We need to load the metadata to calculate weights based on frequency
    logger.info("Calculating class weights...")
    df_train = pd.read_csv(Config.TRAIN_META_PATH)
    class_weights = get_class_weights(df_train)

    # 2. Setup Model, Criterion, Optimizer, Scheduler
    model = AnimalModel(pretrained=True)
    model.to(device)

    criterion = FocalLoss(
        weight=class_weights, gamma=Config.FOCAL_LOSS_GAMMA, reduction="mean"
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 3. Setup EMA
    ema_model = None
    if Config.USE_EMA:
        logger.info("Initializing Model EMA...")
        ema_model = ModelEMA(model, decay=Config.EMA_DECAY, device=device)

    # 4. Training Loop
    best_f1 = -1.0
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss, train_f1 = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            ema_model=ema_model,
        )

        # Validate
        # If EMA is used, we validate the EMA model as it will be used for inference
        val_model = ema_model.ema if ema_model else model
        val_loss, val_f1 = validate(val_model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step()

        # Checkpointing and Early Stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            logger.info(f"New best model found! F1: {best_f1}. Saving checkpoint...")
            torch.save(val_model.state_dict(), Config.BEST_MODEL_PATH)
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            logger.info("Early stopping triggered.")
            break

    logger.info(f"Training completed. Best Validation F1: {best_f1}")
    return best_f1
