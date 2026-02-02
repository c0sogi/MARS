import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import AverageMeter, save_checkpoint, set_seed, get_device
from library.model import NarrowSEResNet
from library.dataset import CactusDataset, get_transforms


def train_one_epoch(train_loader, model, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        train_loader (DataLoader): DataLoader for training data.
        model (nn.Module): The neural network model.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        device (torch.device): Device to run training on.
        epoch (int): Current epoch number.

    Returns:
        dict: Dictionary containing average loss and accuracy for the epoch.
    """
    model.train()

    losses = AverageMeter()
    accuracies = AverageMeter()

    start_time = time.time()

    for i, (images, labels, _) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        # Model outputs logits, shape [Batch, 1]
        outputs = model(images)

        # Ensure labels match output shape [Batch, 1]
        labels = labels.unsqueeze(1)

        loss = criterion(outputs, labels)

        # Compute accuracy (threshold 0.5 on sigmoid probability)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        acc = (preds == labels).float().mean()

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        losses.update(loss.item(), images.size(0))
        accuracies.update(acc.item(), images.size(0))

    epoch_time = time.time() - start_time

    print(
        f"Epoch [{epoch}/{Config.EPOCHS}] Training - "
        f"Time: {epoch_time:.2f}s, "
        f"Loss: {losses.avg:.6f}, "
        f"Acc: {accuracies.avg:.6f}"
    )

    return {"loss": losses.avg, "accuracy": accuracies.avg}


def validate(val_loader, model, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        val_loader (DataLoader): DataLoader for validation data.
        model (nn.Module): The neural network model.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run evaluation on.

    Returns:
        dict: Dictionary containing validation loss, accuracy, and ROC AUC.
    """
    model.eval()

    losses = AverageMeter()

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Ensure labels match output shape [Batch, 1] for loss calculation
            labels_unsqueezed = labels.unsqueeze(1)
            loss = criterion(outputs, labels_unsqueezed)

            losses.update(loss.item(), images.size(0))

            # Store for metric calculation
            probs = torch.sigmoid(outputs)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy().flatten())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate Metrics
    # Accuracy at 0.5 threshold
    all_preds = (all_probs > 0.5).astype(float)
    acc = accuracy_score(all_labels, all_preds)

    # ROC AUC
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        # Handle case where only one class is present in batch (unlikely for full val set)
        auc = 0.5

    print(
        f"Validation - "
        f"Loss: {losses.avg:.10f}, "
        f"Acc: {acc:.10f}, "
        f"AUC: {auc:.10f}"
    )

    return {"loss": losses.avg, "accuracy": acc, "auc": auc}


def train_model(seed, debug=False):
    """
    Manages the training loop for a single model instance (seed).

    Args:
        seed (int): Random seed for this training run.
        debug (bool): If True, runs on a subset of data for debugging.
    """
    # 1. Setup Reproducibility
    set_seed(seed)
    device = get_device()
    print(f"\nStarting training for Seed {seed} on device: {device}")

    # 2. Prepare Data
    train_dataset = CactusDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        split="train",
        transform=get_transforms("train"),
        load_cached_data=True,
        debug=debug,
    )

    val_dataset = CactusDataset(
        metadata_path=Config.VAL_METADATA_PATH,
        split="val",
        transform=get_transforms("val"),
        load_cached_data=True,
        debug=debug,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Initialize Model, Criterion, Optimizer, Scheduler
    model = NarrowSEResNet().to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # 4. Training Loop with Early Stopping
    best_auc = 0.0
    patience = 5
    patience_counter = 0

    model_save_path = os.path.join(Config.WORK_DIR, f"model_seed_{seed}.pth")

    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_metrics = train_one_epoch(
            train_loader, model, criterion, optimizer, device, epoch
        )

        # Validate
        val_metrics = validate(val_loader, model, criterion, device)

        # Step Scheduler
        scheduler.step()

        # Checkpoint & Early Stopping
        current_auc = val_metrics["auc"]

        if current_auc > best_auc:
            best_auc = current_auc
            patience_counter = 0
            print(
                f"New best AUC: {best_auc:.10f}. Saving model to {model_save_path}..."
            )
            save_checkpoint(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            print(f"AUC did not improve. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Finished training for Seed {seed}. Best AUC: {best_auc:.10f}")
