import torch
import torch.nn as nn
import torch.optim as optim
import os
from library.config import Config
from library.utils import seed_everything, get_device, MaskedL1Loss
from library.dataset import get_data_loaders
from library.model import WSDHNet, predict


def train_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    """
    Trains the model for one epoch.

    Args:
        model: The neural network model.
        loader: DataLoader for training data.
        optimizer: Optimizer instance.
        criterion: Loss function.
        device: Torch device.
        scheduler: Learning rate scheduler (optional).

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for x, u_out, y in loader:
        x = x.to(device)
        u_out = u_out.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = model(x)

        # Calculate masked loss
        loss = criterion(preds, y, u_out)

        # Backward pass
        loss.backward()

        # Gradient Clipping (Mandatory)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD_NORM)

        # Optimizer step
        optimizer.step()

        # Scheduler step (OneCycleLR updates per step)
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        count += 1

    return running_loss / count if count > 0 else 0.0


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The neural network model.
        loader: DataLoader for validation data.
        criterion: Loss function.
        device: Torch device.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    running_loss = 0.0
    count = 0

    with torch.no_grad():
        for x, u_out, y in loader:
            x = x.to(device)
            u_out = u_out.to(device)
            y = y.to(device)

            preds = model(x)
            loss = criterion(preds, y, u_out)

            running_loss += loss.item()
            count += 1

    return running_loss / count if count > 0 else 0.0


def run_training(load_cached_data=True):
    """
    Orchestrates the training pipeline, including initialization, data loading,
    model training, and inference.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from cache.
    """
    # 1. Initialization
    Config.initialize()
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Device selected: {device}")

    # 2. Data Loading
    # get_data_loaders handles the caching logic internally via prepare_data
    train_loader, val_loader, test_loader = get_data_loaders(
        load_cached_data=load_cached_data
    )

    # Determine input dimension dynamically from a batch
    sample_x, _, _ = next(iter(train_loader))
    input_dim = sample_x.shape[2]
    print(f"Detected input dimension: {input_dim}")

    # 3. Model Setup
    model = WSDHNet(input_dim=input_dim).to(device)

    # 4. Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Using OneCycleLR as requested
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=10000.0,
    )

    criterion = MaskedL1Loss()

    # 5. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, scheduler
        )
        val_loss = validate(model, val_loader, criterion, device)

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} - Train Loss: {train_loss} - Val Loss: {val_loss}"
        )

        # Early Stopping and Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"New best model saved with Val Loss: {best_val_loss}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered after {patience_counter} epochs.")
                break

    print(f"Training complete. Best Validation Loss: {best_val_loss}")

    # 6. Inference
    # predict function handles loading the best model state and saving submission
    predict(model, test_loader, device)
