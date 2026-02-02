import os
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import (
    LEARNING_RATE,
    WEIGHT_DECAY,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    WORKING_DIR,
)


def train_one_epoch(model, dataloader, optimizer, device, criterion):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The PyTorch model to train.
        dataloader (DataLoader): DataLoader for training data.
        optimizer (Optimizer): Optimizer instance.
        device (torch.device): Device to run training on.
        criterion (nn.Module): Loss function.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in dataloader:
        kin_seq = batch["kinematic_sequence"].to(device)
        ctx_feats = batch["context_features"].to(device)
        targets = batch["target_residual"].to(device)
        batch_size = kin_seq.size(0)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(kin_seq, ctx_feats)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def evaluate(model, dataloader, device, criterion):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The PyTorch model to evaluate.
        dataloader (DataLoader): DataLoader for validation data.
        device (torch.device): Device to run evaluation on.
        criterion (nn.Module): Loss function.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    with torch.no_grad():
        for batch in dataloader:
            kin_seq = batch["kinematic_sequence"].to(device)
            ctx_feats = batch["context_features"].to(device)
            targets = batch["target_residual"].to(device)
            batch_size = kin_seq.size(0)

            outputs = model(kin_seq, ctx_feats)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def fit(model, train_loader, val_loader, device, epochs, patience):
    """
    Runs the full training loop with early stopping and scheduler.

    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        device (torch.device): Computation device.
        epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.

    Returns:
        nn.Module: The trained model with best weights loaded.
    """
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        verbose=True,
    )
    criterion = nn.L1Loss()  # Mean Absolute Error

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)
        val_loss = evaluate(model, val_loader, device, criterion)

        # Update learning rate
        scheduler.step(val_loss)

        # Print metrics with full precision
        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss}"
        )

        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    print(f"Best Validation Loss: {best_val_loss}")

    # Load best model weights
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print("Loaded best model weights.")

    return model
