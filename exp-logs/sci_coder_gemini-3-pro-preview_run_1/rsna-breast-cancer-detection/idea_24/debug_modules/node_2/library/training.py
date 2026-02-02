import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from library.config import Config
from library.utils import get_logger, pf1_score, seed_everything
from library.data_loader import get_dataloaders
from library.model import AsymmetryGatedSiameseNetwork

logger = get_logger("training")


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """
    Training logic for one epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    # Iterate over batches
    # Using tqdm for progress tracking if run interactively, though prompt asks to minimize printing.
    # We will iterate directly.
    for batch_idx, (target_img, contra_img, labels) in enumerate(loader):
        # Move to device
        target_img = target_img.to(device)
        contra_img = contra_img.to(device)
        labels = labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        # Model expects: target_img, contra_img
        logits = model(target_img, contra_img)

        # Flatten logits to match label shape (B,)
        logits = logits.view(-1)

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Optimizer step
        # NOTE: Gradient clipping is EXPLICITLY DISABLED as per strategy
        # to allow large updates for the minority class.
        optimizer.step()

        # Update metrics
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        count += batch_size

    avg_loss = running_loss / count if count > 0 else 0.0
    return avg_loss


def validate_epoch(model, loader, criterion, device):
    """
    Validation logic. Computes Loss and pF1 score.
    """
    model.eval()
    running_loss = 0.0
    count = 0

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (target_img, contra_img, labels) in enumerate(loader):
            target_img = target_img.to(device)
            contra_img = contra_img.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(target_img, contra_img)
            logits = logits.view(-1)

            # Loss
            loss = criterion(logits, labels)

            # Accumulate for metrics
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            # Update running loss
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            count += batch_size

    avg_loss = running_loss / count if count > 0 else 0.0

    # Concatenate all predictions and labels
    if len(all_probs) > 0:
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        # Compute pF1
        val_pf1 = pf1_score(all_labels, all_probs)
    else:
        val_pf1 = 0.0

    return avg_loss, val_pf1


def run_training(
    num_epochs=Config.NUM_EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    pos_weight=Config.POS_WEIGHT,
    debug=Config.DEBUG,
    debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
    patience=3,
    save_path=None,
):
    """
    Main function to run the training pipeline.

    Args:
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        learning_rate (float): Learning rate for AdamW.
        pos_weight (float): Positive class weight for BCE loss.
        debug (bool): Whether to run in debug mode (subsampled data).
        debug_sample_size (int): Number of samples in debug mode.
        patience (int): Early stopping patience.
        save_path (str): Path to save the best model. If None, uses default cache dir.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    if save_path is None:
        save_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    # 1. Data Loading
    logger.info("Initializing DataLoaders...")
    train_loader, val_loader, _ = get_dataloaders(
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        load_cached_data=True,
        debug=debug,
        debug_sample_size=debug_sample_size,
    )

    # 2. Model Initialization
    logger.info("Initializing Model...")
    model = AsymmetryGatedSiameseNetwork()
    model = model.to(device)

    # 3. Loss Function
    # Weighted BCE to handle class imbalance (Pos Weight ~ 47.0)
    pos_weight_tensor = torch.tensor([pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # 4. Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    # 5. Training Loop
    logger.info("Starting training...")
    best_val_loss = float("inf")
    early_stop_counter = 0

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_pf1 = validate_epoch(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Logging (Full precision as requested)
        logger.info(f"Epoch {epoch}/{num_epochs}")
        logger.info(f"Train Loss: {train_loss}")
        logger.info(f"Validation Loss: {val_loss}")
        logger.info(f"Validation pF1: {val_pf1}")
        logger.info(f"Learning Rate: {current_lr}")

        # Checkpoint & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            logger.info(f"Validation loss improved. Saving model to {save_path}")
            torch.save(model.state_dict(), save_path)
        else:
            early_stop_counter += 1
            logger.info(
                f"No improvement. Early stopping counter: {early_stop_counter}/{patience}"
            )

        if early_stop_counter >= patience:
            logger.info("Early stopping triggered.")
            break

    logger.info(f"Training complete. Best Validation Loss: {best_val_loss}")
    return model
