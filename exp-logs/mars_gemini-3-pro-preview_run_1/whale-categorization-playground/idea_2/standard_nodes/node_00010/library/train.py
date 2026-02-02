import os
import time
import torch
import torch.nn as nn
import numpy as np

from library.config import Config
from library.utils import AverageMeter, map5, set_seed
from library.data import get_loaders
from library.model import WhaleEfficientNet


def train_one_epoch(
    loader, model, criterion, optimizer, device, epoch, accumulation_steps=2
):
    """
    Trains the model for one epoch using gradient accumulation.
    """
    model.train()
    losses = AverageMeter()

    # Zero gradients at the start of the epoch
    optimizer.zero_grad()

    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        logits = model(images)
        loss = criterion(logits, labels)

        # Scale loss by accumulation steps
        loss = loss / accumulation_steps

        # Backward pass
        loss.backward()

        # Update metrics (multiply back to log correct loss value)
        losses.update(loss.item() * accumulation_steps, images.size(0))

        # Optimizer step only after accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    # Handle any remaining gradients if loader size is not divisible by accumulation_steps
    if len(loader) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return losses.avg


def validate(loader, model, criterion, device):
    """
    Evaluates the model on the validation set.
    Computes Loss and MAP@5.
    """
    model.eval()
    losses = AverageMeter()

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass with TTA (Horizontal Flip)
            logits1 = model(images)
            logits2 = model(torch.flip(images, dims=[3]))
            logits = (logits1 + logits2) / 2.0

            loss = criterion(logits, labels)

            losses.update(loss.item(), images.size(0))

            # Get Top 5 predictions
            # logits shape: (Batch, NumClasses)
            # indices shape: (Batch, 5)
            _, top5_indices = logits.topk(5, dim=1, largest=True, sorted=True)

            # Collect targets and preds for MAP@5 calculation
            # We use indices directly as the metric function handles scalar vs list comparison correctly
            all_targets.extend(labels.cpu().numpy().tolist())
            all_preds.extend(top5_indices.cpu().numpy().tolist())

    # Calculate MAP@5
    # targets is a list of scalars (int class indices)
    # preds is a list of lists (top 5 int class indices)
    val_map5 = map5(all_targets, all_preds)

    return losses.avg, val_map5


def run_training(debug=False, epochs=None, patience=5):
    """
    Main function to run the training pipeline.

    Args:
        debug (bool): If True, runs on a subset of data.
        epochs (int): Number of epochs to train. If None, uses Config.EPOCHS.
        patience (int): Early stopping patience.
    """
    # 1. Setup
    set_seed(Config.SEED)
    Config.setup_directories()

    device = Config.DEVICE
    if epochs is None:
        epochs = Config.EPOCHS

    print(f"Starting training on device: {device}")
    print(f"Debug Mode: {debug}")
    print(f"Epochs: {epochs}")

    # 2. Data Loaders
    # load_cached_data=True allows using the cached classes.npy if available
    train_loader, val_loader = get_loaders(debug=debug, load_cached_data=True)

    # 3. Model
    model = WhaleEfficientNet(pretrained=True)
    model = model.to(device)

    # 4. Loss, Optimizer, Scheduler
    # Using Label Smoothing to prevent overfitting on singleton classes
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.SCHEDULER_T_MAX, eta_min=Config.SCHEDULER_MIN_LR
    )

    # 5. Training Loop
    best_map5 = 0.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            device,
            epoch,
            accumulation_steps=2,
        )

        # Validate
        val_loss, val_map5 = validate(val_loader, model, criterion, device)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - start_time

        # Print Metrics (Full Precision)
        print(f"Epoch {epoch}/{epochs} | Time: {elapsed:.2f}s | LR: {current_lr:.8f}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val MAP@5: {val_map5}")

        # Checkpoint & Early Stopping
        if val_map5 > best_map5:
            best_map5 = val_map5
            patience_counter = 0
            print(f"New best MAP@5! Saving checkpoint to {Config.CHECKPOINT_PATH}")
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_map5": best_map5,
                    "optimizer": optimizer.state_dict(),
                },
                Config.CHECKPOINT_PATH,
            )
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best MAP@5: {best_map5}")
