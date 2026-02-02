import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from library.config import Config
from library.utils import set_seed, AverageMeter, save_checkpoint, calculate_roc_auc
from library.model import ModifiedDenseNet
from library.dataset import get_dataloaders


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Executes one training epoch.

    Args:
        model: The neural network model.
        loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to run training on (cpu or cuda).

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    losses = AverageMeter()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # Ensure shape (B, 1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The neural network model.
        loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to run evaluation on.

    Returns:
        tuple: (Average validation loss, Validation AUC score)
    """
    model.eval()
    losses = AverageMeter()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            losses.update(loss.item(), images.size(0))

            # Apply sigmoid to convert logits to probabilities
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate AUC
    auc = calculate_roc_auc(all_targets, all_preds)

    return losses.avg, auc


def run_training(debug=False):
    """
    Main function to run the training pipeline.

    Args:
        debug (bool): If True, runs with a smaller subset of data for debugging.
    """
    # Set reproducibility
    set_seed(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Training on device: {device}")

    # Load Data
    train_loader, val_loader, _ = get_dataloaders(debug=debug)

    # Initialize Model
    model = ModifiedDenseNet(pretrained=True)
    model = model.to(device)

    # Define Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Define Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    # Training Loop Variables
    best_auc = 0.0
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train Phase
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validation Phase
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Update Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        elapsed = time.time() - start_time

        # Print Metrics (Full Precision)
        print(
            f"Epoch {epoch + 1}/{Config.EPOCHS} | Time: {elapsed:.2f}s | LR: {current_lr}"
        )
        print(f"  Train Loss: {train_loss}")
        print(f"  Val Loss:   {val_loss}")
        print(f"  Val AUC:    {val_auc}")

        # Early Stopping & Checkpointing
        if val_auc > best_auc:
            print(
                f"  Validation AUC improved from {best_auc} to {val_auc}. Saving model..."
            )
            best_auc = val_auc
            patience_counter = 0
            save_checkpoint(model.state_dict(), Config.MODEL_PATH)
        else:
            patience_counter += 1
            print(
                f"  Validation AUC did not improve. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation AUC: {best_auc}")
