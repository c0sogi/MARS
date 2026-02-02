import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import get_logger, save_checkpoint

# Initialize logger
logger = get_logger("engine")


def train_one_epoch(model, loader, optimizer, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): The training data loader.
        optimizer (Optimizer): The optimizer.
        device (torch.device): The device to run on.
        epoch (int): Current epoch number.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    criterion = nn.CrossEntropyLoss()

    for batch_idx, (images, labels, _) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    logger.info(f"Epoch {epoch} Training Loss: {epoch_loss}")
    return epoch_loss


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): The validation data loader.
        device (torch.device): The device to run on.

    Returns:
        float: Average CrossEntropy Loss (Log Loss).
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

    avg_loss = running_loss / dataset_size
    # Print full precision as requested
    logger.info(f"Validation Loss: {avg_loss}")
    return avg_loss


def predict(model, loader, device):
    """
    Generates predictions for the dataset.

    Args:
        model (nn.Module): The model to use for inference.
        loader (DataLoader): The data loader (test or val).
        device (torch.device): The device to run on.

    Returns:
        np.ndarray: Array of probabilities with shape (n_samples, n_classes).
        list: List of IDs corresponding to the predictions.
    """
    model.eval()
    all_probs = []
    all_ids = []

    with torch.no_grad():
        for images, _, ids in loader:
            images = images.to(device)

            # Forward pass
            outputs = model(images)

            # Apply Softmax to get probabilities
            probs = torch.softmax(outputs, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_ids.extend(ids)

    return np.concatenate(all_probs), all_ids


def train_fold(fold_idx, model, train_loader, val_loader, device):
    """
    Orchestrates the training process for a single fold, including warm-up and fine-tuning.

    Args:
        fold_idx (int): Index of the current fold.
        model (nn.Module): The model instance.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        device (torch.device): Device to compute on.

    Returns:
        float: Best validation loss achieved.
    """
    logger.info(f"Starting training for Fold {fold_idx}")

    best_loss = float("inf")
    # Early stopping patience (allow full convergence as per idea, but stop if degrading significantly)
    patience = 7
    patience_counter = 0

    # Define checkpoint path for this fold
    checkpoint_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold_idx}.pth")
    best_model_path = os.path.join(
        Config.WORKING_DIR, f"best_model_fold_{fold_idx}.pth"
    )

    # ==========================================
    # Phase 1: Warm-up
    # ==========================================
    if Config.WARMUP_EPOCHS > 0:
        logger.info("Phase 1: Warm-up (Frozen Backbone)")
        model.set_backbone_trainable(False)

        # Use a standard LR for head initialization, slightly higher than fine-tuning
        optimizer_warmup = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3,
            weight_decay=Config.WEIGHT_DECAY,
        )

        for epoch in range(1, Config.WARMUP_EPOCHS + 1):
            logger.info(f"Warm-up Epoch {epoch}/{Config.WARMUP_EPOCHS}")
            train_one_epoch(model, train_loader, optimizer_warmup, device, epoch)
            val_loss = validate(model, val_loader, device)

            # Save state after warmup
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "loss": val_loss,
                },
                checkpoint_path,
            )

    # ==========================================
    # Phase 2: Fine-tuning
    # ==========================================
    logger.info("Phase 2: Fine-tuning (Unfrozen Backbone)")
    model.set_backbone_trainable(True)

    # Re-initialize optimizer for all parameters with conservative LR
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR
    )

    for epoch in range(1, Config.FINE_TUNE_EPOCHS + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Fine-tune Epoch {epoch}/{Config.FINE_TUNE_EPOCHS} | LR: {current_lr:.2e}"
        )

        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = validate(model, val_loader, device)

        scheduler.step()

        # Checkpoint logic
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            patience_counter = 0
            logger.info(f"New best model found! Loss: {best_loss}")
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "loss": best_loss,
                },
                best_model_path,
            )
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{patience}")

        # Always save latest state
        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "loss": val_loss,
            },
            checkpoint_path,
        )

        # Early Stopping
        if patience_counter >= patience:
            logger.info("Early stopping triggered.")
            break

    logger.info(f"Fold {fold_idx} finished. Best Validation Loss: {best_loss}")
    return best_loss
