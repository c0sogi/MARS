import os
import time
import torch
import numpy as np
import math
from library.config import DEVICE, CHECKPOINT_DIR, MAX_GRAD_NORM
from library.utils import seed_everything


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(model, optimizer, data_loader, criterion, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model: The neural network model.
        optimizer: The optimizer.
        data_loader: PyTorch DataLoader for training data.
        criterion: The loss function.
        device: Compute device (cpu or cuda).
        epoch: Current epoch number.

    Returns:
        dict: Average metrics for the epoch.
    """
    model.train()

    # Trackers for different loss components
    losses = AverageMeter()
    hm_losses = AverageMeter()
    size_losses = AverageMeter()
    off_losses = AverageMeter()
    glob_losses = AverageMeter()

    for batch_idx, (images, targets, _, _) in enumerate(data_loader):
        images = images.to(device)

        # Move targets to device
        targets = {k: v.to(device) for k, v in targets.items()}

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss, loss_dict = criterion(outputs, targets)

        # Check for NaN
        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training")
            print(loss_dict)
            return None

        # Backward pass
        loss.backward()

        # Gradient Clipping (Critical for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)

        # Optimizer step
        optimizer.step()

        # Update metrics
        batch_size = images.size(0)
        losses.update(loss.item(), batch_size)
        hm_losses.update(loss_dict["hm_loss"], batch_size)
        size_losses.update(loss_dict["size_loss"], batch_size)
        off_losses.update(loss_dict["off_loss"], batch_size)
        glob_losses.update(loss_dict["glob_loss"], batch_size)

    return {
        "loss": losses.avg,
        "hm_loss": hm_losses.avg,
        "size_loss": size_losses.avg,
        "off_loss": off_losses.avg,
        "glob_loss": glob_losses.avg,
    }


def evaluate(model, data_loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The neural network model.
        data_loader: PyTorch DataLoader for validation data.
        criterion: The loss function.
        device: Compute device.

    Returns:
        dict: Average metrics for the validation set.
    """
    model.eval()

    losses = AverageMeter()
    hm_losses = AverageMeter()
    size_losses = AverageMeter()
    off_losses = AverageMeter()
    glob_losses = AverageMeter()

    with torch.no_grad():
        for images, targets, _, _ in data_loader:
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss, loss_dict = criterion(outputs, targets)

            # Update metrics
            batch_size = images.size(0)
            losses.update(loss.item(), batch_size)
            hm_losses.update(loss_dict["hm_loss"], batch_size)
            size_losses.update(loss_dict["size_loss"], batch_size)
            off_losses.update(loss_dict["off_loss"], batch_size)
            glob_losses.update(loss_dict["glob_loss"], batch_size)

    return {
        "val_loss": losses.avg,
        "val_hm_loss": hm_losses.avg,
        "val_size_loss": size_losses.avg,
        "val_off_loss": off_losses.avg,
        "val_glob_loss": glob_losses.avg,
    }


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    epochs,
    device=DEVICE,
    patience=5,
    save_path=CHECKPOINT_DIR,
):
    """
    Main training loop with Early Stopping and Checkpointing.

    Args:
        model: Model to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        criterion: Loss function.
        epochs: Total number of epochs.
        device: Device to train on.
        patience: Epochs to wait before early stopping.
        save_path: Directory to save checkpoints.
    """
    os.makedirs(save_path, exist_ok=True)
    best_loss = float("inf")
    early_stopping_counter = 0

    print(f"Starting training on device: {device}")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # --- Training ---
        train_metrics = train_one_epoch(
            model, optimizer, train_loader, criterion, device, epoch
        )

        if train_metrics is None:
            print("Training failed due to NaN loss.")
            break

        # --- Validation ---
        val_metrics = evaluate(model, val_loader, criterion, device)

        # --- Scheduling ---
        if scheduler is not None:
            # If scheduler is ReduceLROnPlateau, it needs a metric
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["val_loss"])
            else:
                scheduler.step()

        # --- Logging ---
        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch}/{epochs} | Time: {elapsed:.2f}s | LR: {current_lr}")
        print(
            f"Train Loss: {train_metrics['loss']} (HM: {train_metrics['hm_loss']}, Size: {train_metrics['size_loss']}, Off: {train_metrics['off_loss']}, Glob: {train_metrics['glob_loss']})"
        )
        print(
            f"Val Loss: {val_metrics['val_loss']} (HM: {val_metrics['val_hm_loss']}, Size: {val_metrics['val_size_loss']}, Off: {val_metrics['val_off_loss']}, Glob: {val_metrics['val_glob_loss']})"
        )

        # --- Checkpointing & Early Stopping ---
        current_val_loss = val_metrics["val_loss"]

        # Save Last Model
        last_model_path = os.path.join(save_path, "last_model.pth")
        torch.save(model.state_dict(), last_model_path)

        if current_val_loss < best_loss:
            best_loss = current_val_loss
            early_stopping_counter = 0
            best_model_path = os.path.join(save_path, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with Val Loss: {best_loss}")
        else:
            early_stopping_counter += 1
            print(
                f"No improvement. Early stopping counter: {early_stopping_counter}/{patience}"
            )

        if early_stopping_counter >= patience:
            print("Early stopping triggered.")
            break

    print("Training complete.")
