import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from library.config import Config, set_seed
from library.utils import get_logger, log_metrics
from library.data_loader import get_dataloaders
from library.model import AsymmetricEfficientNet

# Initialize logger
logger = get_logger(name="Training")


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)  # Ensure shape (N, 1)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)

        # Store predictions (sigmoid) and targets for AUC
        all_targets.append(targets.detach().cpu().numpy())
        all_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    # Concatenate all batches
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    # robust AUC calculation
    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        # Handle cases where batch might have only one class
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())

    val_loss = running_loss / len(loader.dataset)

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    try:
        val_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        val_auc = 0.5

    return val_loss, val_auc


def run_training(epochs=Config.EPOCHS, debug=False):
    """
    Main function to orchestrate the training process.
    """
    set_seed(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting training on device: {device}")

    # 1. Load Metadata
    if not os.path.exists(Config.TRAIN_METADATA) or not os.path.exists(
        Config.VAL_METADATA
    ):
        logger.error(
            "Metadata files not found. Ensure metadata generation was successful."
        )
        return

    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)

    if debug:
        logger.info(
            f"Debug mode enabled. Truncating datasets to {Config.MAX_DEBUG_SAMPLES} samples."
        )
        train_df = train_df.iloc[: Config.MAX_DEBUG_SAMPLES]
        val_df = val_df.iloc[: Config.MAX_DEBUG_SAMPLES]

    # 2. Prepare DataLoaders
    loaders = get_dataloaders(train_df=train_df, val_df=val_df)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # 3. Initialize Model
    model = AsymmetricEfficientNet()
    model = model.to(device)

    # 4. Optimizer & Loss
    # Using AdamW with moderate weight decay as per strategy
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    # BCEWithLogitsLoss combines Sigmoid + BCE, numerically stable
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_auc = 0.0
    patience_counter = 0

    logger.info("Starting training loop...")

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        # Log Metrics
        metrics = {
            "Epoch": epoch + 1,
            "Train Loss": train_loss,
            "Train AUC": train_auc,
            "Val Loss": val_loss,
            "Val AUC": val_auc,
            "Time (s)": elapsed,
        }
        log_metrics(metrics, logger)

        # Checkpointing (Maximize AUC)
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            logger.info(f"New best model saved! AUC: {best_auc}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            logger.info(
                f"Early stopping triggered. No improvement for {patience_counter} epochs."
            )
            break

    logger.info(f"Training finished. Best Validation AUC: {best_auc}")
