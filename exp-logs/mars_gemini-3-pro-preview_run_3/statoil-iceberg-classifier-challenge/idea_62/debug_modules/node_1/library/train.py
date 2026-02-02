import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from library.config import Config
from library.utils import set_seed, save_checkpoint, AverageMeter
from library.data_loader import get_fold_loaders
from library.model import LSEIsomorphicCNN


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    losses = AverageMeter()
    model.train()

    for images, angles, targets in loader:
        # Move data to device
        images = images.to(device)
        angles = angles.to(device)
        targets = targets.to(device).unsqueeze(1)  # Ensure shape is (B, 1)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, angles)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update metrics
        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    losses = AverageMeter()
    model.eval()

    with torch.no_grad():
        for images, angles, targets in loader:
            images = images.to(device)
            angles = angles.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(images, angles)
            loss = criterion(outputs, targets)

            losses.update(loss.item(), images.size(0))

    return losses.avg


def run_cross_validation():
    """
    Executes the 5-Fold Cross-Validation training pipeline.
    """
    # Setup environment (directories, seeds)
    Config.setup()
    device = torch.device(Config.DEVICE)

    print(f"Starting {Config.N_FOLDS}-Fold Cross-Validation")
    print(f"Device: {device}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Learning Rate: {Config.LEARNING_RATE}")

    fold_best_losses = []

    for fold_idx in range(Config.N_FOLDS):
        print(f"\n========== Fold {fold_idx} ==========")

        # Reset seed for each fold to ensure consistent model initialization
        set_seed(Config.SEED)

        # Get DataLoaders for this fold
        train_loader, val_loader = get_fold_loaders(fold_idx, load_cached_data=True)

        # Initialize Model
        model = LSEIsomorphicCNN().to(device)

        # Initialize Optimizer (AdamW)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Loss Function
        criterion = nn.BCEWithLogitsLoss()

        # Training Loop Variables
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(Config.NUM_EPOCHS):
            # Train
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )

            # Validate
            val_loss = validate(model, val_loader, criterion, device)

            # Print Metrics (Full Precision)
            print(
                f"Epoch {epoch + 1}/{Config.NUM_EPOCHS} | Train Loss: {train_loss} | Val Loss: {val_loss}"
            )

            # Checkpoint Logic
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            # Save Checkpoint
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "fold": fold_idx,
                },
                is_best,
                Config.CHECKPOINT_DIR,
                fold_idx,
            )

            # Early Stopping
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        print(f"Fold {fold_idx} Best Validation Loss: {best_val_loss}")
        fold_best_losses.append(best_val_loss)

    # Summary
    print("\n========== CV Summary ==========")
    for i, loss in enumerate(fold_best_losses):
        print(f"Fold {i}: {loss}")

    avg_loss = sum(fold_best_losses) / len(fold_best_losses)
    print(f"Average Validation Loss: {avg_loss}")
