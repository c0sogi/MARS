import torch
import numpy as np
import os
from library.utils import AverageMeter
from library.config import Config


def train_fn(dataloader, model, criterion, optimizer, device, scheduler=None):
    """
    Executes one epoch of training.

    Args:
        dataloader: PyTorch DataLoader for training data.
        model: The neural network model.
        criterion: Loss function (e.g., CrossEntropyLoss).
        optimizer: Optimizer instance.
        device: Device to run on (cuda/cpu).
        scheduler: Learning rate scheduler (optional).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    for i, data in enumerate(dataloader):
        images = data["image"].to(device)
        targets = data["target"].to(device)

        optimizer.zero_grad()

        # Forward pass:
        # We pass labels to the model because the SubCenterArcFace head
        # requires them to calculate the angular margin logits.
        outputs = model(images, labels=targets)

        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))

        # Note: CosineAnnealingLR is typically stepped once per epoch in the loop,
        # not per batch. If a batch-level scheduler is used, step here.

    return loss_meter.avg


def eval_fn(dataloader, model, criterion, device):
    """
    Executes validation for one epoch.

    Args:
        dataloader: PyTorch DataLoader for validation data.
        model: The neural network model.
        criterion: Loss function.
        device: Device to run on.

    Returns:
        float: Average loss for the epoch.
    """
    model.eval()
    loss_meter = AverageMeter()

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            images = data["image"].to(device)
            targets = data["target"].to(device)

            # We pass labels to compute the loss on the ArcFace logits.
            # This helps track if the model is converging on the metric learning objective.
            outputs = model(images, labels=targets)

            loss = criterion(outputs, targets)
            loss_meter.update(loss.item(), images.size(0))

    return loss_meter.avg


def extract_embeddings(dataloader, model, device):
    """
    Extracts feature embeddings for the entire dataloader.
    Used for Gallery Construction and Query Expansion.

    Args:
        dataloader: PyTorch DataLoader.
        model: The neural network model.
        device: Device to run on.

    Returns:
        np.ndarray: Array of embeddings with shape (N_samples, Embedding_Dim).
    """
    model.eval()
    embeddings = []

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            images = data["image"].to(device)

            # Forward pass without labels returns the embeddings (features)
            # from the backbone + neck, skipping the head.
            emb = model(images)
            embeddings.append(emb.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def train_loop(
    train_loader,
    val_loader,
    model,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs=Config.EPOCHS,
    patience=5,
):
    """
    Orchestrates the full training process with Early Stopping.

    Args:
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        model: Model to train.
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: LR Scheduler.
        device: Device.
        epochs: Total number of epochs.
        patience: Epochs to wait for improvement before early stopping.
    """
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # Training Step
        train_loss = train_fn(
            train_loader, model, criterion, optimizer, device, scheduler
        )

        # Validation Step
        val_loss = eval_fn(val_loader, model, criterion, device)

        # Scheduler Step (CosineAnnealingLR is stepped epoch-wise)
        if scheduler is not None:
            scheduler.step()

        # Log metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss}"
        )

        # Early Stopping & Model Saving
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            # Also save to the generic path to ensure a model exists there
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
