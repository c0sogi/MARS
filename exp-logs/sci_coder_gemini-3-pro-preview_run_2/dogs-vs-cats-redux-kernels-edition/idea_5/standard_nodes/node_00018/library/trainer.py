import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import AverageMeter, mixup_data, mixup_criterion, save_checkpoint


def train_one_epoch(train_loader, model, criterion, optimizer, device):
    """
    Executes one epoch of training.
    """
    model.train()
    losses = AverageMeter()

    for i, (images, target) in enumerate(train_loader):
        images = images.to(device)
        target = target.to(device)

        # Apply Mixup augmentation
        images, targets_a, targets_b, lam = mixup_data(
            images, target, Config.mixup_alpha, device
        )

        # Forward pass
        # Model outputs [Batch, 1], squeeze to [Batch] to match target shape
        output = model(images).squeeze(1)

        # Compute Mixup loss
        loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate_one_epoch(val_loader, model, criterion, device):
    """
    Executes validation for one epoch.
    """
    model.eval()
    losses = AverageMeter()

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # Forward pass
            output = model(images).squeeze(1)

            # Compute standard loss (Log Loss)
            loss = criterion(output, target)

            # Update metrics
            losses.update(loss.item(), images.size(0))

    return losses.avg


def train_fold(fold_idx, train_loader, val_loader, model):
    """
    Orchestrates the training process for a single fold.
    """
    device = Config.device

    # Loss function: BCEWithLogitsLoss combines Sigmoid and BCELoss for numerical stability
    # This directly optimizes Log Loss.
    criterion = nn.BCEWithLogitsLoss().to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.epochs, eta_min=Config.min_lr
    )

    best_loss = float("inf")

    print(f"Starting training for Fold {fold_idx}...")

    for epoch in range(Config.epochs):
        # Run training epoch
        train_loss = train_one_epoch(train_loader, model, criterion, optimizer, device)

        # Run validation epoch
        val_loss = validate_one_epoch(val_loader, model, criterion, device)

        # Update learning rate
        scheduler.step()

        # Print metrics with full precision
        print(
            f"Fold {fold_idx} | Epoch {epoch + 1}/{Config.epochs} | Train Loss: {train_loss} | Val Loss: {val_loss}"
        )

        # Save checkpoint if validation loss improves
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_loss": best_loss,
                    "optimizer": optimizer.state_dict(),
                    "fold": fold_idx,
                },
                Config.checkpoint_dir,
                f"fold_{fold_idx}.pth",
            )
            print(f"Saved best model for Fold {fold_idx} with Val Loss: {best_loss}")

    return best_loss
