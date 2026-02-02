import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from library.config import Config
from library.utils import get_logger, seed_everything
from library.data import BrainTumorDataset
from library.model import AsymmetricEfficientNet

# Initialize logger
logger = get_logger("train")


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device).float().view(-1, 1)

        optimizer.zero_grad()

        # Forward pass (model outputs logits)
        logits = model(inputs)

        # Calculate loss
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC score.
    """
    model.eval()
    running_loss = 0.0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().view(-1, 1)

            logits = model(inputs)
            loss = criterion(logits, labels)

            running_loss += loss.item() * inputs.size(0)

            # Apply sigmoid to convert logits to probabilities for metric calculation
            probs = torch.sigmoid(logits)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    # Calculate ROC AUC
    # Handle edge case where only one class is present in the batch/dataset
    try:
        auc_score = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc_score = 0.5

    return epoch_loss, auc_score


def run_training():
    """
    Main function to orchestrate the training process.
    """
    seed_everything(Config.SEED)

    logger.info(f"Starting training with device: {Config.DEVICE}")

    # 1. Load Metadata
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)

    logger.info(f"Train samples: {len(df_train)}")
    logger.info(f"Val samples: {len(df_val)}")

    # 2. Initialize Datasets and Dataloaders
    train_dataset = BrainTumorDataset(df_train, phase="train", load_cached_data=True)
    val_dataset = BrainTumorDataset(df_val, phase="val", load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Initialize Model, Loss, Optimizer
    model = AsymmetricEfficientNet()
    model = model.to(Config.DEVICE)

    # BCEWithLogitsLoss includes Sigmoid, numerically more stable
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 4. Training Loop with Early Stopping
    best_auc = 0.0
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{Config.EPOCHS}")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE
        )
        val_loss, val_auc = validate(model, val_loader, criterion, Config.DEVICE)

        # Print metrics with full precision
        print(f"Epoch {epoch + 1} - Train Loss: {train_loss}")
        print(f"Epoch {epoch + 1} - Val Loss: {val_loss}")
        print(f"Epoch {epoch + 1} - Val AUC: {val_auc}")

        # Checkpoint and Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            logger.info(f"New best AUC found: {best_auc}. Saving model...")
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
        else:
            patience_counter += 1
            logger.info(
                f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            logger.info("Early stopping triggered.")
            break

    logger.info(f"Training complete. Best Validation AUC: {best_auc}")
