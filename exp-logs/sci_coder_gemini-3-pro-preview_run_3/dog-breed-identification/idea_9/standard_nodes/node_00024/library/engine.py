import os
import torch
import torch.nn as nn
from library.config import Config
from library.model import freeze_backbone, unfreeze_backbone


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """
    Trains the model for one epoch using CrossEntropyLoss.

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): Training data loader.
        optimizer (Optimizer): The optimizer.
        device (torch.device): Device to run on.
        epoch (int): Current epoch number (for logging/scheduling).

    Returns:
        tuple: (average_loss, average_accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # Metrics
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        _, predicted = torch.max(outputs.data, 1)
        total += batch_size
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def evaluate(model, dataloader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): Validation data loader.
        device (torch.device): Device to run on.

    Returns:
        tuple: (average_loss, average_accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            # Metrics
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs.data, 1)
            total += batch_size
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def train_loop(model, train_loader, val_loader, optimizer, scheduler, device, fold_idx):
    """
    Orchestrates the training process including Phase 1 (Linear Probing),
    Phase 2 (Fine-Tuning), Early Stopping, and SWA Checkpointing.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data.
        val_loader (DataLoader): Validation data.
        optimizer (Optimizer): Optimizer.
        scheduler (LRScheduler): Learning rate scheduler.
        device (torch.device): Computation device.
        fold_idx (int): Current fold index for file naming.
    """
    # ==========================================
    # Phase 1: Linear Probing
    # ==========================================
    print(f"\nFold {fold_idx}: Starting Phase 1 (Linear Probing)")
    freeze_backbone(model)

    # Set Phase 1 Learning Rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = Config.PHASE1_LR

    for epoch in range(Config.PHASE1_EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, epoch
        )
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(
            f"Phase 1 Epoch {epoch + 1}: "
            f"Train Loss {train_loss}, Train Acc {train_acc}, "
            f"Val Loss {val_loss}, Val Acc {val_acc}"
        )

    # ==========================================
    # Phase 2: Fine-Tuning
    # ==========================================
    print(f"\nFold {fold_idx}: Starting Phase 2 (Fine-Tuning)")
    unfreeze_backbone(model)

    # Reset Learning Rate for Phase 2
    for param_group in optimizer.param_groups:
        param_group["lr"] = Config.LEARNING_RATE

    best_loss = float("inf")
    patience = 5  # Early stopping patience
    counter = 0

    for epoch in range(Config.EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, epoch
        )
        val_loss, val_acc = evaluate(model, val_loader, device)

        if scheduler:
            scheduler.step()

        print(
            f"Phase 2 Epoch {epoch + 1}: "
            f"Train Loss {train_loss}, Train Acc {train_acc}, "
            f"Val Loss {val_loss}, Val Acc {val_acc}"
        )

        # Save Best Model (Standard Checkpoint)
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(
                model.state_dict(),
                os.path.join(Config.WORKING_DIR, f"best_model_fold_{fold_idx}.pth"),
            )
        else:
            counter += 1

        # SWA Checkpointing: Save models from the last N epochs
        if epoch >= (Config.EPOCHS - Config.SWA_EPOCHS):
            swa_path = os.path.join(
                Config.WORKING_DIR, f"swa_fold_{fold_idx}_epoch_{epoch}.pth"
            )
            torch.save(model.state_dict(), swa_path)

        # Early Stopping
        # We only stop early if we are NOT in the SWA region.
        # If we are in the SWA region, we continue to collect checkpoints for averaging.
        if counter >= patience and epoch < (Config.EPOCHS - Config.SWA_EPOCHS):
            print(f"Early stopping triggered at Phase 2 Epoch {epoch + 1}")
            break
