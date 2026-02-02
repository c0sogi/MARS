import os
import torch
import numpy as np
import pandas as pd
from library.utils import AverageMeter, calculate_roc_auc, save_checkpoint


def train_one_epoch(model, dataloader, optimizer, device, loss_fn, scheduler=None):
    """
    Trains the model for one epoch.

    Args:
        model: PyTorch model.
        dataloader: PyTorch DataLoader.
        optimizer: PyTorch Optimizer.
        device: Device to train on.
        loss_fn: Loss function (DistillationLoss).
        scheduler: Learning rate scheduler (optional).
    """
    model.train()
    loss_meter = AverageMeter()

    for batch in dataloader:
        # Determine batch contents based on dataset mode
        if len(batch) == 3:
            # Distillation mode: image, label, teacher_logits
            images, labels, teacher_logits = batch
            images = images.to(device)
            labels = labels.to(device)
            teacher_logits = teacher_logits.to(device)
        else:
            # Standard mode: image, label
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            teacher_logits = None

        optimizer.zero_grad()

        # Forward pass
        logits = model(images)

        # Calculate loss
        if teacher_logits is not None:
            loss = loss_fn(logits, labels, teacher_logits)
        else:
            loss = loss_fn(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        loss_meter.update(loss.item(), images.size(0))

    return loss_meter.avg


def validate(model, dataloader, device, loss_fn=None):
    """
    Evaluates the model on the validation set.

    Args:
        model: PyTorch model.
        dataloader: PyTorch DataLoader.
        device: Device to evaluate on.
        loss_fn: Loss function (optional).

    Returns:
        avg_loss: Average validation loss.
        score: ROC AUC score.
        logits: Raw logits (numpy).
        labels: True labels (numpy).
    """
    model.eval()
    loss_meter = AverageMeter()

    logits_list = []
    labels_list = []

    with torch.no_grad():
        for batch in dataloader:
            # Handle potential 3-tuple if validation set uses distillation mode
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            if loss_fn:
                # Validation loss
                try:
                    loss = loss_fn(logits, labels, teacher_logits=None)
                except TypeError:
                    loss = loss_fn(logits, labels)
                loss_meter.update(loss.item(), images.size(0))

            logits_list.append(logits.cpu())
            labels_list.append(labels.cpu())

    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)

    # Calculate Probabilities
    preds = torch.softmax(logits, dim=1)

    # Calculate Score
    score = calculate_roc_auc(labels, preds)

    return loss_meter.avg, score, logits.numpy(), labels.numpy()


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    loss_fn,
    cfg,
    scheduler=None,
    patience=5,
    save_path=None,
):
    """
    Main training loop with Early Stopping.

    Args:
        model: PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader (can be None).
        optimizer: Optimizer.
        device: Device.
        loss_fn: Loss function.
        cfg: Config object.
        scheduler: Scheduler.
        patience: Early stopping patience.
        save_path: Path to save best model.

    Returns:
        best_score: Best validation score.
    """
    best_score = -float("inf")
    patience_counter = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, loss_fn, scheduler
        )

        val_info = ""
        if val_loader:
            val_loss, val_score, _, _ = validate(model, val_loader, device, loss_fn)
            val_info = f" Val Loss: {val_loss} Val AUC: {val_score}"

            # Checkpointing & Early Stopping
            if save_path:
                if val_score > best_score:
                    best_score = val_score
                    patience_counter = 0
                    save_checkpoint(model, optimizer, epoch, val_score, save_path)
                else:
                    patience_counter += 1
        else:
            # No validation (Full training)
            # Save every epoch to ensure we have the latest state
            if save_path:
                save_checkpoint(model, optimizer, epoch, 0.0, save_path)

        # Step Scheduler (Epoch-based)
        if scheduler:
            scheduler.step()

        print(f"Epoch {epoch} Train Loss: {train_loss}{val_info}")

        if val_loader and patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    return best_score


def inference(model, dataloader, device):
    """
    Performs inference on the test set.

    Args:
        model: PyTorch model.
        dataloader: PyTorch DataLoader (Test mode).
        device: Device.

    Returns:
        ids: List of image IDs.
        preds: Predicted probabilities (numpy).
    """
    model.eval()

    ids_list = []
    logits_list = []

    with torch.no_grad():
        for images, ids in dataloader:
            images = images.to(device)

            logits = model(images)

            logits_list.append(logits.cpu())
            ids_list.extend(ids)

    logits = torch.cat(logits_list)
    preds = torch.softmax(logits, dim=1).numpy()

    return ids_list, preds


def generate_submission(model, dataloader, device, save_path, target_cols):
    """
    Generates and saves the submission file.

    Args:
        model: PyTorch model.
        dataloader: Test DataLoader.
        device: Device.
        save_path: Path to save CSV.
        target_cols: List of target column names.
    """
    ids, preds = inference(model, dataloader, device)

    df_sub = pd.DataFrame({"image_id": ids})

    for i, col in enumerate(target_cols):
        df_sub[col] = preds[:, i]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_sub.to_csv(save_path, index=False)
