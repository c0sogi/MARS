import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import AverageMeter, save_checkpoint, set_seed
from library.model import SimpleCNN
from library.data_loader import get_loaders


def train_one_epoch(train_loader, model, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch.
    """
    losses = AverageMeter()
    model.train()

    for i, (images, angles, targets) in enumerate(train_loader):
        images = images.to(device)
        angles = angles.to(device)
        targets = targets.to(device).view(-1, 1)

        # Forward pass
        outputs = model(images, angles)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(val_loader, model, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    losses = AverageMeter()
    model.eval()

    with torch.no_grad():
        for i, (images, angles, targets) in enumerate(val_loader):
            images = images.to(device)
            angles = angles.to(device)
            targets = targets.to(device).view(-1, 1)

            # Forward pass
            outputs = model(images, angles)
            loss = criterion(outputs, targets)

            # Update metrics
            losses.update(loss.item(), images.size(0))

    return losses.avg


def run_training_fold(fold_idx):
    """
    Runs the training and validation loop for a specific fold.
    """
    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Setup directories
    fold_dir = os.path.join(Config.WORK_DIR, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    checkpoint_path = os.path.join(fold_dir, "checkpoint.pth")

    print(f"Starting training for Fold {fold_idx}...")

    # Device configuration
    device = torch.device(Config.DEVICE)

    # Data Loaders
    train_loader, val_loader = get_loaders(fold_idx=fold_idx, load_cached_data=True)

    # Model Initialization
    model = SimpleCNN(drop_rate=Config.DROP_RATE, fc_dim=Config.FC_DIM)
    model = model.to(device)

    # Loss and Optimizer
    # BCEWithLogitsLoss is used because the model outputs raw logits (no Sigmoid at the end)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Training Loop
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            train_loader, model, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss = validate(val_loader, model, criterion, device)

        elapsed = time.time() - start_time

        print(
            f"Fold {fold_idx} | Epoch [{epoch+1}/{Config.NUM_EPOCHS}] | "
            f"Train Loss: {train_loss} | Val Loss: {val_loss} | "
            f"Time: {elapsed:.2f}s"
        )

        # Checkpoint and Early Stopping
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            patience_counter = 0
            print(
                f"New best model found for Fold {fold_idx} with Val Loss: {best_loss}"
            )
        else:
            patience_counter += 1

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_loss": best_loss,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            filename=checkpoint_path,
        )

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered for Fold {fold_idx} at epoch {epoch+1}")
            break

    print(f"Fold {fold_idx} finished. Best Validation Loss: {best_loss}")
    return best_loss
