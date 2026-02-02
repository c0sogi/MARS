import os
import torch
import numpy as np
from sklearn.metrics import f1_score
from library.config import Config
from library.utils import MetricMonitor
from library.ema import ModelEMA


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    scaler,
    ema_model=None,
    max_batches=None,
):
    """
    Trains the model for one epoch using AMP and optional EMA updates.

    Args:
        model: The PyTorch model.
        train_loader: DataLoader for training data.
        optimizer: The optimizer.
        criterion: The loss function.
        device: The device to run on.
        scaler: GradScaler for AMP.
        ema_model: Optional ModelEMA instance.
        max_batches: Optional integer to limit the number of batches (for debugging).

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    metric_monitor = MetricMonitor()

    for batch_idx, (images, targets) in enumerate(train_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Automatic Mixed Precision context
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if ema_model:
            ema_model.update(model)

        metric_monitor.update("Loss", loss.item())

    return metric_monitor.avg["Loss"]


def evaluate(model, val_loader, criterion, device, max_batches=None):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        val_loader: DataLoader for validation data.
        criterion: The loss function.
        device: The device to run on.
        max_batches: Optional integer to limit the number of batches.

    Returns:
        tuple: (Average Loss, Mean F1-Score)
    """
    model.eval()
    metric_monitor = MetricMonitor()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            metric_monitor.update("Loss", loss.item())

            # Apply sigmoid activation
            probs = torch.sigmoid(outputs)
            # Apply threshold to get binary predictions
            preds = (probs > Config.CONF_THRESHOLD).float()

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        # Calculate Macro F1-Score
        f1 = f1_score(all_targets, all_preds, average="macro")
    else:
        f1 = 0.0

    return metric_monitor.avg["Loss"], f1


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    num_epochs=Config.EPOCHS,
    max_batches_per_epoch=None,
):
    """
    Main training loop with Early Stopping and Checkpointing.

    Args:
        model: The PyTorch model.
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        optimizer: The optimizer.
        criterion: The loss function.
        device: The device.
        num_epochs: Total number of epochs.
        max_batches_per_epoch: Limit batches per epoch for debugging.

    Returns:
        float: Best Validation F1 Score achieved.
    """
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Initialize EMA if configured
    ema_model = None
    if Config.USE_EMA:
        ema_model = ModelEMA(model, decay=Config.EMA_DECAY, device=device)

    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    best_f1 = -1.0
    patience = 5
    patience_counter = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            ema_model,
            max_batches=max_batches_per_epoch,
        )

        # Validate
        # Use EMA weights for validation if available, else standard weights
        val_model = ema_model.module if ema_model else model
        val_loss, val_f1 = evaluate(
            val_model, val_loader, criterion, device, max_batches=max_batches_per_epoch
        )

        # Step Scheduler
        scheduler.step()

        # Print metrics (Full precision)
        print(
            f"Epoch: {epoch} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val F1: {val_f1}"
        )

        # Checkpointing & Early Stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(val_model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    return best_f1
