import os
import torch
import numpy as np
from library.utils import MetricMonitor


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Executes one training epoch.

    Args:
        model: The PyTorch model.
        train_loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to run on (cuda/cpu).
        epoch: Current epoch number.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    metric_monitor = MetricMonitor()

    for batch_idx, (images, masks, depths, ids) in enumerate(train_loader):
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)
        depths = depths.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        # Forward pass: Model expects (x, z)
        outputs = model(images, depths)

        # Calculate loss
        loss = criterion(outputs, masks)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        metric_monitor.update("Loss", loss.item())

    print(f"Epoch: {epoch} | Train | {metric_monitor}")
    return metric_monitor.avg["Loss"]


def validate_one_epoch(model, val_loader, criterion, device, epoch):
    """
    Executes one validation epoch.

    Args:
        model: The PyTorch model.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to run on.
        epoch: Current epoch number.

    Returns:
        tuple: (Average Loss, Predictions (logits), Targets, IDs)
    """
    model.eval()
    metric_monitor = MetricMonitor()

    predictions = []
    targets = []
    ids_list = []

    with torch.no_grad():
        for batch_idx, (images, masks, depths, ids) in enumerate(val_loader):
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)
            depths = depths.to(device, dtype=torch.float32)

            outputs = model(images, depths)
            loss = criterion(outputs, masks)

            metric_monitor.update("Loss", loss.item())

            # Collect logits and targets for threshold optimization
            # Move to CPU to conserve GPU memory
            predictions.append(outputs.cpu().numpy())
            targets.append(masks.cpu().numpy())
            ids_list.extend(ids)

    print(f"Epoch: {epoch} | Val | {metric_monitor}")

    # Concatenate collected data
    if len(predictions) > 0:
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
    else:
        predictions = np.array([])
        targets = np.array([])

    return metric_monitor.avg["Loss"], predictions, targets, ids_list


def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs,
    patience,
    save_dir,
):
    """
    Runs the full training pipeline with Early Stopping.

    Args:
        model: The PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Device.
        epochs: Maximum number of epochs.
        patience: Early stopping patience.
        save_dir: Directory to save the best model.

    Returns:
        float: Best validation loss achieved.
    """
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.pth")

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_preds, val_targets, val_ids = validate_one_epoch(
            model, val_loader, criterion, device, epoch
        )

        # Step Scheduler
        if scheduler is not None:
            scheduler.step()

        # Print full precision loss
        print(f"Epoch {epoch} Validation Loss: {val_loss}")

        # Early Stopping Logic
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path} with loss: {best_loss}")
        else:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    return best_loss
