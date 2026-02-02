import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from library.config import Config
from library.utils import seed_everything, get_logger, pf1_score
from library.dataset import get_dataloaders
from library.model import SiameseFPNEfficientNet

# Initialize logger
logger = get_logger("train")


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Training loop for one epoch.
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        # Move data to device
        images = batch["image"].to(device)
        images_contra = batch["image_contra"].to(device)
        labels = batch["label"].to(device)

        # Forward pass
        # The model accepts paired inputs for the Siamese architecture
        logits = model(images, images_contra)

        # Ensure labels are correct shape for BCEWithLogitsLoss (N, 1)
        labels = labels.unsqueeze(1)

        loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # NOTE: Gradient clipping is explicitly DISABLED as per strategy
        # to allow large updates from the weighted loss function.
        if Config.MAX_GRAD_NORM is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        # Track metrics
        running_loss += loss.item() * images.size(0)

        # Store for epoch-level pF1 calculation (optional, but good for monitoring)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.append(probs)
        all_labels.append(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    # Calculate training pF1
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    epoch_pf1 = pf1_score(all_labels, all_preds)

    return epoch_loss, epoch_pf1


def validate(model, loader, criterion, device):
    """
    Validation loop.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            images_contra = batch["image_contra"].to(device)
            labels = batch["label"].to(device)

            logits = model(images, images_contra)
            labels = labels.unsqueeze(1)

            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(labels.cpu().numpy())

    total_loss = running_loss / len(loader.dataset)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Calculate Probabilistic F1
    pf1 = pf1_score(all_labels, all_preds)

    return total_loss, pf1


def run_training(load_cached_data=True):
    """
    Main execution function for training the model.
    """
    seed_everything(Config.SEED)

    # Create working directory
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    device = torch.device(Config.DEVICE)
    logger.info(f"Using device: {device}")

    # 1. Data Loading
    logger.info("Initializing DataLoaders...")
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=load_cached_data)

    # 2. Model Initialization
    logger.info("Initializing Siamese FPN Model...")
    model = SiameseFPNEfficientNet()
    model.to(device)

    # 3. Loss Function
    # Aggressive positive weighting to handle 1:47 imbalance
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # 5. Training Loop
    best_pf1 = -1.0

    logger.info("Starting training...")
    logger.info(f"Epochs: {Config.EPOCHS}")
    logger.info(f"Batch Size: {Config.BATCH_SIZE}")
    logger.info(f"Positive Weight: {Config.POS_WEIGHT}")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss, train_pf1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Validate
        val_loss, val_pf1 = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Logging
        logger.info(f"Epoch {epoch+1}/{Config.EPOCHS}")
        logger.info(f"  LR: {current_lr:.2e}")
        logger.info(f"  Train Loss: {train_loss} | Train pF1: {train_pf1}")
        logger.info(f"  Val Loss:   {val_loss} | Val pF1:   {val_pf1}")

        # Checkpointing (Maximize pF1)
        if val_pf1 > best_pf1:
            logger.info(
                f"  [Improvement] pF1 increased from {best_pf1} to {val_pf1}. Saving model..."
            )
            best_pf1 = val_pf1
            torch.save(model.state_dict(), Config.MODEL_PATH)
        else:
            logger.info(f"  [No Improvement] Best pF1 remains {best_pf1}")

    logger.info("Training complete.")
    logger.info(f"Best Validation pF1: {best_pf1}")
    return best_pf1
