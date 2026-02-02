import os
import time
import gc
import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from library.config import Config
from library.utils import seed_everything, get_score, get_logger
from library.data import get_loaders
from library.model import CatheterModel
from library.loss import CustomLoss


def train_one_epoch(model, optimizer, criterion, dataloader, device, epoch, scaler):
    """
    Trains the model for one epoch using Gradient Accumulation and Mixed Precision.
    """
    model.train()

    running_loss = 0.0
    dataset_size = 0

    # Gradient Accumulation Config
    accumulation_steps = Config.GRADIENT_ACCUMULATION_STEPS

    optimizer.zero_grad()

    for step, (images, labels, masks) in enumerate(dataloader):
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)

        batch_size = images.size(0)

        # Mixed Precision Forward Pass
        with autocast():
            logits, mask_preds = model(images)
            loss = criterion(logits, mask_preds, labels, masks)
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps

        # Backward Pass with Scaler
        scaler.scale(loss).backward()

        # Optimization Step (only every accumulation_steps or at end of epoch)
        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
            # Unscale gradients for clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

            # Step optimizer and updater scaler
            scaler.step(optimizer)
            scaler.update()

            # Flush gradients
            optimizer.zero_grad()

        # Accumulate metrics (multiply by accumulation_steps to get back original loss magnitude for logging)
        running_loss += (loss.item() * accumulation_steps) * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def valid_one_epoch(model, criterion, dataloader, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and AUC score.
    """
    model.eval()

    running_loss = 0.0
    dataset_size = 0

    # Store predictions and targets for AUC calculation
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels, masks in dataloader:
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            batch_size = images.size(0)

            # Forward Pass (No Autocast needed for validation usually, but consistent behavior is good)
            # Using float32 for stable evaluation
            logits, mask_preds = model(images)
            loss = criterion(logits, mask_preds, labels, masks)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to logits for probability
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    epoch_auc = get_score(all_targets, all_preds)

    return epoch_loss, epoch_auc


def run_training():
    """
    Main training pipeline.
    """
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)
    logger = get_logger("train")
    device = torch.device(Config.DEVICE)

    logger.info(f"Device: {device}")
    logger.info(f"Gradient Accumulation Steps: {Config.GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"Effective Batch Size: {Config.EFFECTIVE_BATCH_SIZE}")

    # 2. Data Loaders
    logger.info("Loading data...")
    train_loader, val_loader = get_loaders(
        train_batch_size=Config.BATCH_SIZE,
        val_batch_size=Config.BATCH_SIZE,
        load_cached_data=True,
        debug=Config.DEBUG,
    )

    # 3. Model, Loss, Optimizer
    logger.info("Initializing model...")
    model = CatheterModel(pretrained=Config.PRETRAINED)
    model.to(device)

    criterion = CustomLoss(
        cls_weight=Config.CLS_LOSS_WEIGHT, aux_weight=Config.AUX_LOSS_WEIGHT
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR
    )

    scaler = GradScaler()

    # 4. Training Loop
    best_auc = 0.0
    patience = 3
    patience_counter = 0

    logger.info("Starting training...")

    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, optimizer, criterion, train_loader, device, epoch, scaler
        )

        # Validation
        val_loss, val_auc = valid_one_epoch(model, criterion, val_loader, device)

        # Scheduler Step
        scheduler.step()

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} "
            f"[Time: {elapsed:.0f}s] "
            f"[LR: {current_lr:.2e}] "
            f"Train Loss: {train_loss:.6f} "
            f"Val Loss: {val_loss:.6f} "
            f"Val AUC: {val_auc}"
        )

        # Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            logger.info(f"Validation AUC Improved ({best_auc}). Saving model...")
            torch.save(model.state_dict(), Config.MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{patience}")

        # Early Stopping
        if patience_counter >= patience:
            logger.info("Early stopping triggered.")
            break

        # Memory Cleanup
        gc.collect()
        torch.cuda.empty_cache()

    logger.info(f"Training complete. Best AUC: {best_auc}")


if __name__ == "__main__":
    run_training()
