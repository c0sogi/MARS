import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import log_loss, accuracy_score
from library.utils import get_logger

# Initialize logger
logger = get_logger("engine")


def train_one_epoch(model, loader, optimizer, device, epoch, mixup_fn=None):
    """
    Trains the model for one epoch.

    Args:
        model: PyTorch model.
        loader: DataLoader for training data.
        optimizer: Optimizer.
        device: 'cuda' or 'cpu'.
        epoch: Current epoch number.
        mixup_fn: Optional timm Mixup function.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    # BCEWithLogitsLoss is used for binary classification with a single output logit
    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)

        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)
            # timm Mixup with num_classes=2 returns (Batch, 2) one-hot encoded targets.
            # Our model outputs (Batch, 1). We need the probability of class 1.
            # Extract the second column (index 1) which corresponds to the 'Dog' class probability.
            targets = targets[:, 1].view(-1, 1)
        else:
            # If no mixup, ensure targets are float and correct shape
            targets = targets.float().view(-1, 1)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        dataset_size += images.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(model, loader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: PyTorch model.
        loader: DataLoader for validation data.
        device: 'cuda' or 'cpu'.

    Returns:
        dict: Dictionary containing loss, log_loss, accuracy, preds, and targets.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device).float().view(-1, 1)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            dataset_size += images.size(0)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    avg_loss = running_loss / dataset_size

    # Calculate metrics
    # Log Loss (primary metric)
    ll = log_loss(all_targets, all_preds)
    # Accuracy (threshold 0.5)
    acc = accuracy_score(all_targets, (all_preds > 0.5).astype(int))

    return {
        "loss": avg_loss,
        "log_loss": ll,
        "accuracy": acc,
        "preds": all_preds,
        "targets": all_targets,
    }


def predict_tta(model, loader, device):
    """
    Generates predictions using Test-Time Augmentation (Horizontal Flip).

    Args:
        model: PyTorch model.
        loader: DataLoader for test data (returns image, id).
        device: 'cuda' or 'cpu'.

    Returns:
        tuple: (predictions numpy array, ids numpy array)
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in loader:
            images, img_ids = batch
            images = images.to(device)

            # 1. Forward pass on original images
            out_orig = model(images)
            prob_orig = torch.sigmoid(out_orig)

            # 2. Forward pass on horizontally flipped images
            # Flip on width dimension (dim 3 for NCHW)
            images_flipped = torch.flip(images, dims=[3])
            out_flip = model(images_flipped)
            prob_flip = torch.sigmoid(out_flip)

            # 3. Average probabilities
            prob_avg = (prob_orig + prob_flip) / 2.0

            all_preds.append(prob_avg.cpu().numpy())
            all_ids.append(img_ids.numpy())

    # Flatten predictions to 1D array
    all_preds = np.concatenate(all_preds).flatten()
    all_ids = np.concatenate(all_ids).flatten()

    return all_preds, all_ids


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs,
    mixup_fn,
    config,
    fold,
    model_name,
    patience=20,
):
    """
    Orchestrates the training loop, including logging, checkpointing, and early stopping.

    Args:
        model: PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Device.
        epochs: Total epochs.
        mixup_fn: Mixup function.
        config: Config object.
        fold: Current fold index.
        model_name: Name of the model architecture.
        patience: Early stopping patience.

    Returns:
        float: Best validation log loss achieved.
    """
    best_log_loss = float("inf")
    patience_counter = 0

    # Path for the best model (based on validation metric)
    best_model_path = os.path.join(
        config.checkpoint_dir, f"best_{model_name}_fold_{fold}.pth"
    )

    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, mixup_fn
        )

        # Validate
        val_metrics = evaluate(model, val_loader, device)

        # Step Scheduler
        if scheduler is not None:
            scheduler.step()

        # Log Metrics
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Fold {fold} | Epoch {epoch+1}/{epochs} | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_metrics['loss']:.6f} | "
            f"Val LogLoss: {val_metrics['log_loss']:.6f} | "
            f"Val Acc: {val_metrics['accuracy']:.6f}"
        )

        # Save Checkpoints for Model Soup
        # We save the last `soup_epochs` to average them later
        if epoch >= epochs - config.soup_epochs:
            soup_ckpt_path = os.path.join(
                config.checkpoint_dir, f"{model_name}_fold_{fold}_epoch_{epoch}.pth"
            )
            torch.save(model.state_dict(), soup_ckpt_path)
            # logger.info(f"Saved soup checkpoint: {soup_ckpt_path}")

        # Early Stopping & Best Model Logic
        if val_metrics["log_loss"] < best_log_loss:
            best_log_loss = val_metrics["log_loss"]
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved to {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

    return best_log_loss
