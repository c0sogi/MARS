import os
import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import AverageMeter, calculate_auc, get_logger
from library.data import get_loaders
from library.model import HybridEfficientNet


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, logger):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses = AverageMeter()

    # Iterate over batches
    for step, batch in enumerate(loader):
        images = batch["image"].to(device)
        tabular = batch["tabular"].to(device)
        targets = batch["target"].to(device).unsqueeze(1)  # [Batch, 1]

        optimizer.zero_grad()

        # Forward pass
        logits = model(images, tabular)

        # Calculate loss
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    logger.info(f"Epoch {epoch+1} - Train Loss: {losses.avg:.6f}")
    return losses.avg


def validate_one_epoch(model, loader, criterion, device, logger):
    """
    Validates the model on the validation set.
    """
    model.eval()
    losses = AverageMeter()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for step, batch in enumerate(loader):
            images = batch["image"].to(device)
            tabular = batch["tabular"].to(device)
            targets = batch["target"].to(device).unsqueeze(1)

            # Forward pass
            logits = model(images, tabular)
            loss = criterion(logits, targets)

            losses.update(loss.item(), images.size(0))

            # Apply sigmoid to get probabilities
            preds = torch.sigmoid(logits)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    # Concatenate all batches
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    # Calculate AUC
    auc_score = calculate_auc(all_targets, all_preds)

    logger.info(f"Validation Loss: {losses.avg:.6f}")
    logger.info(f"Validation AUC: {auc_score}")  # Full precision print

    return losses.avg, auc_score


def run_fold(fold, load_cached_data=True):
    """
    Runs the training and validation loop for a specific fold.
    Saves the best model based on Validation AUC.
    """
    # Setup Logger
    log_file = os.path.join(Config.WORKING_DIR, f"train_fold_{fold}.log")
    logger = get_logger(f"Fold_{fold}", log_file)
    logger.info(f"Starting Fold {fold}")

    # Device
    device = torch.device(Config.DEVICE)

    # Data Loaders
    train_loader, val_loader, num_tabular_features = get_loaders(
        fold, load_cached_data=load_cached_data
    )

    # Model Initialization
    model = HybridEfficientNet(
        model_name=Config.MODEL_NAME,
        pretrained=True,
        num_classes=Config.NUM_CLASSES,
        num_tabular_features=num_tabular_features,
        tabular_hidden_dim=Config.TABULAR_HIDDEN_DIM,
        final_dropout=Config.FINAL_DROPOUT,
    )
    model.to(device)

    # Loss Function (Weighted BCE)
    # Using dampened positive weight as per strategy
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer (AdamW)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Training Loop
    best_auc = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, f"fold_{fold}_best.pth")

    for epoch in range(Config.EPOCHS):
        logger.info(f"--- Epoch {epoch+1}/{Config.EPOCHS} ---")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, logger
        )

        # Validate
        val_loss, val_auc = validate_one_epoch(
            model, val_loader, criterion, device, logger
        )

        # Checkpoint based on AUC
        if val_auc > best_auc:
            logger.info(f"AUC Improved ({best_auc} -> {val_auc}). Saving model...")
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
        else:
            logger.info(f"AUC did not improve (Best: {best_auc})")

    logger.info(f"Fold {fold} Finished. Best AUC: {best_auc}")
    return best_auc
