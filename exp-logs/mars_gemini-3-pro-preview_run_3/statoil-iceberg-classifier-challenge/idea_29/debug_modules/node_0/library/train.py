import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from library import config, utils, model, data_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch_idx, (images, angles, labels) in enumerate(loader):
        # Move data to device
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        # Model expects (x, inc_angle)
        outputs = model(images, angles)

        # Calculate loss
        # BCEWithLogitsLoss expects target shape (N, *) same as input (N, 1)
        loss = criterion(outputs, labels.unsqueeze(1))

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / num_batches


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, (images, angles, labels) in enumerate(loader):
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device)

            outputs = model(images, angles)
            loss = criterion(outputs, labels.unsqueeze(1))

            running_loss += loss.item()
            num_batches += 1

    return running_loss / num_batches


def train_fold(fold_index):
    """
    Orchestrates the training process for a single fold.
    Includes initialization, training loop, validation, early stopping, and checkpointing.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fold {fold_index}: Starting training on device {device}")

    # Get DataLoaders
    # We rely on the caching mechanism in data_loader
    train_loader, val_loader, _, _ = data_loader.get_loaders(
        fold_index, load_cached_data=True
    )

    # Initialize Model
    net = model.EAP_CNN()
    net = net.to(device)

    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Early Stopping variables
    best_val_loss = float("inf")
    patience_counter = 0

    start_time = time.time()

    for epoch in range(config.NUM_EPOCHS):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(net, train_loader, criterion, optimizer, device)

        # Validate
        val_loss = validate(net, val_loader, criterion, device)

        epoch_duration = time.time() - epoch_start

        print(
            f"Fold {fold_index} | Epoch {epoch+1}/{config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss} | Val Loss: {val_loss} | "
            f"Time: {epoch_duration:.2f}s"
        )

        # Checkpoint and Early Stopping
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            print(
                f"Fold {fold_index}: New best model found (Val Loss: {best_val_loss}). Saving checkpoint."
            )
        else:
            patience_counter += 1

        # Save checkpoint
        utils.save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": net.state_dict(),
                "best_val_loss": best_val_loss,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            fold_index,
            config.CHECKPOINT_DIR,
        )

        if patience_counter >= config.PATIENCE:
            print(
                f"Fold {fold_index}: Early stopping triggered after {epoch+1} epochs."
            )
            break

    total_time = time.time() - start_time
    print(
        f"Fold {fold_index}: Training complete. Best Val Loss: {best_val_loss}. Total time: {total_time:.2f}s"
    )

    return best_val_loss
