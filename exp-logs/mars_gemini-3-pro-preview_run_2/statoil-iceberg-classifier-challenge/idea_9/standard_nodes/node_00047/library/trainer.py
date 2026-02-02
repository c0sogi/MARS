import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.model import A2SHN
from library.data_loader import get_data_loaders
from library.utils import seed_everything, save_checkpoint, log_metrics


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network.
        dataloader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        device (str): Device to run on ('cpu' or 'cuda').

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for images, angles, labels in dataloader:
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device).unsqueeze(1)  # Ensure shape (Batch, 1)

        optimizer.zero_grad()

        outputs = model(images, angles)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / num_batches if num_batches > 0 else 0.0


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network.
        dataloader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        device (str): Device to run on.

    Returns:
        tuple: (Average validation loss, Validation accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    with torch.no_grad():
        for images, angles, labels in dataloader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images, angles)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            num_batches += 1

            # Accuracy calculation (threshold 0.5)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy


def run_fold(fold_idx, load_cached_data=True):
    """
    Runs the training and validation for a specific fold.

    Args:
        fold_idx (int): The fold index (0 to N_FOLDS-1).
        load_cached_data (bool): Whether to use cached data.

    Returns:
        float: The best validation loss achieved in this fold.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Starting training for Fold {fold_idx} on {device}...")

    # 2. Data Loaders
    train_loader, val_loader, _ = get_data_loaders(
        fold=fold_idx, load_cached_data=load_cached_data
    )

    # 3. Model Initialization
    model = A2SHN().to(device)

    # 4. Optimizer & Criterion
    # "Low and Slow" strategy: Adam with conservative LR
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Binary Cross Entropy Loss
    criterion = nn.BCELoss()

    # Scheduler: Reduce LR when validation loss stagnates
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # 5. Training Loop Variables
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    # Path to save the best model for this fold
    save_path = os.path.join(Config.WORK_DIR, f"a2shn_model_fold_{fold_idx}.pth")

    start_time = time.time()

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step(val_loss)

        # Logging
        epoch_duration = time.time() - epoch_start
        log_metrics(epoch, train_loss, val_loss, val_acc, epoch_duration)

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Deep copy the best model state
            best_model_state = copy.deepcopy(model.state_dict())

            # Save to disk immediately to ensure we have the file
            save_checkpoint(model, optimizer, epoch, val_loss, save_path)
            # print(f"  -> New best model saved (Val Loss: {best_val_loss})")
        else:
            patience_counter += 1
            # print(f"  -> No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    total_time = time.time() - start_time
    print(
        f"Fold {fold_idx} finished in {total_time:.2f}s. Best Val Loss: {best_val_loss}"
    )

    return best_val_loss
