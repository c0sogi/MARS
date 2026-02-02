import torch
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import seed_everything, MCRMSELoss, EarlyStopping
from library.data import get_dataloaders
from library.model import RNA_Net


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    return avg_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    return avg_loss


def run_training(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    load_cached_data=True,
    patience=Config.EARLY_STOPPING_PATIENCE,
):
    """
    Main training loop.

    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for data loaders.
        load_cached_data (bool): Whether to load pre-processed data from cache.
        patience (int): Patience for early stopping.
    """
    # Set reproducibility
    seed_everything(Config.SEED)

    device = Config.DEVICE
    print(f"Using device: {device}")

    # Initialize Model
    model = RNA_Net()
    model = model.to(device)

    # Optimizer and Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # Loss Function
    criterion = MCRMSELoss()

    # Early Stopping
    early_stopping = EarlyStopping(
        patience=patience, verbose=True, path=Config.MODEL_PATH
    )

    # Data Loaders
    train_loader, val_loader = get_dataloaders(
        load_cached_data=load_cached_data, batch_size=batch_size
    )

    print("Starting training...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss}"
        )

        # Update Scheduler
        scheduler.step(val_loss)

        # Check Early Stopping
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print("Training complete.")
    print(f"Best model saved to {Config.MODEL_PATH}")
