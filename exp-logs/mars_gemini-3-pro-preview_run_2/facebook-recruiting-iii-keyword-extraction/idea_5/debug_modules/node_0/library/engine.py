import os
import sys
import numpy as np
import torch
import torch.nn as nn

from library.config import Config
from library.model import LinearTaggingModel, FocalLoss
from library.utils import set_seed, Timer


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the training set.
        optimizer (Optimizer): The optimizer to update weights.
        criterion (nn.Module): The loss function.
        device (torch.device): The device (CPU/GPU) to use.
        epoch (int): Current epoch number.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for i, (inputs, targets) in enumerate(dataloader):
        # Move data to device
        # inputs: (batch_size, vocab_size)
        # targets: (batch_size, num_tags)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the validation set.
        criterion (nn.Module): The loss function.
        device (torch.device): The device (CPU/GPU) to use.

    Returns:
        tuple: (average_loss, probabilities, targets)
               probabilities and targets are numpy arrays.
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Forward pass
            logits = model(inputs)
            loss = criterion(logits, targets)

            running_loss += loss.item()
            num_batches += 1

            # Apply sigmoid to get probabilities [0, 1]
            probs = torch.sigmoid(logits)

            # Move to CPU to save GPU memory and accumulate
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0

    # Concatenate all batches into single numpy arrays
    if len(all_probs) > 0:
        all_probs = np.vstack(all_probs)
        all_targets = np.vstack(all_targets)
    else:
        all_probs = np.array([])
        all_targets = np.array([])

    return avg_loss, all_probs, all_targets


def train_model(model, train_loader, val_loader):
    """
    Main training loop with Early Stopping.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.

    Returns:
        nn.Module: The trained model with the best weights loaded.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    model.to(device)

    # Initialize Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Initialize Loss Function
    criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)

    # Early Stopping Variables
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training on {device}...")
    print(f"Training for {Config.EPOCHS} epochs with patience {Config.PATIENCE}.")

    for epoch in range(Config.EPOCHS):
        with Timer(f"Epoch {epoch + 1}"):
            # Train
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device, epoch
            )

            # Validate
            val_loss, _, _ = validate(model, val_loader, criterion, device)

            # Print Metrics (Full Precision)
            print(f"Epoch {epoch + 1} Summary:")
            print(f"Train Loss: {train_loss}")
            print(f"Val Loss: {val_loss}")

            # Early Stopping Logic
            delta = best_val_loss - val_loss
            if delta > Config.MIN_DELTA:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), best_model_path)
                print(f"Validation loss improved. Model saved to {best_model_path}")
            else:
                patience_counter += 1
                print(
                    f"No improvement in validation loss. Patience: {patience_counter}/{Config.PATIENCE}"
                )

            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

    # Load the best model weights before returning
    if os.path.exists(best_model_path):
        print(f"Loading best model weights from {best_model_path}...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: No best model file found. Returning current model.")

    return model
