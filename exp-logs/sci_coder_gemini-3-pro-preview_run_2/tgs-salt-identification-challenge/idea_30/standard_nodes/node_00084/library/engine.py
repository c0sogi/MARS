import os
import torch
import numpy as np
from library.utils import MetricMonitor, calculate_map_vectorized
from library.dataset import ORIG_SIZE, TARGET_SIZE


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

    # --- Calculate mAP with Dynamic Thresholding ---
    # Squeeze channel dim: (N, 1, H, W) -> (N, H, W)
    if predictions.ndim == 4:
        predictions = predictions.squeeze(1)
    if targets.ndim == 4:
        targets = targets.squeeze(1)

    # Crop to original size (101x101)
    pad_t = (TARGET_SIZE - ORIG_SIZE) // 2
    pad_l = (TARGET_SIZE - ORIG_SIZE) // 2

    preds_cropped = predictions[:, pad_t : pad_t + ORIG_SIZE, pad_l : pad_l + ORIG_SIZE]
    targets_cropped = targets[:, pad_t : pad_t + ORIG_SIZE, pad_l : pad_l + ORIG_SIZE]

    # Sigmoid
    probs = 1.0 / (1.0 + np.exp(-preds_cropped))
    targets_bool = targets_cropped > 0.5

    # Search for best threshold
    best_map = 0.0
    thresholds = np.linspace(0.1, 0.9, 17)

    for th in thresholds:
        p_bool = probs > th
        score = calculate_map_vectorized(p_bool, targets_bool)
        if score > best_map:
            best_map = score

    print(f"Epoch {epoch} Best mAP: {best_map:.4f}")

    return metric_monitor.avg["Loss"], predictions, targets, ids_list, best_map


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
    Runs the full training pipeline with Early Stopping based on mAP.
    """
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.pth")

    best_map = 0.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_preds, val_targets, val_ids, val_map = validate_one_epoch(
            model, val_loader, criterion, device, epoch
        )

        # Step Scheduler
        if scheduler is not None:
            scheduler.step()

        # Early Stopping Logic (Maximize mAP)
        if val_map > best_map:
            best_map = val_map
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path} with mAP: {best_map:.4f}")
        else:
            patience_counter += 1
            print(
                f"Early stopping counter: {patience_counter}/{patience} (Best mAP: {best_map:.4f})"
            )

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    return best_map
