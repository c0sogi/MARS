import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import set_seed, calculate_rmse, save_checkpoint, load_checkpoint
from library.model import SRDN
from library.data_loader import get_dataloaders


def train_one_epoch(model, dataloader, optimizer, criterion, device, max_batches=None):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): The training data loader.
        optimizer (Optimizer): The optimizer.
        criterion (nn.Module): The loss function.
        device (torch.device): The device to run training on.
        max_batches (int, optional): Maximum number of batches to process (for debugging).

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for i, (inputs, targets) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass: Model predicts the clean image
        clean_pred = model(inputs)

        # Calculate predicted noise: Noise = Input - Clean Prediction
        noise_pred = inputs - clean_pred

        # Loss is MSE between predicted noise and actual noise target
        loss = criterion(noise_pred, targets)

        loss.backward()

        # Gradient Clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP_VALUE)

        optimizer.step()

        # Accumulate loss
        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    return epoch_loss


def validate(model, dataloader, device, max_batches=None):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): The validation data loader.
        device (torch.device): The device to run evaluation on.
        max_batches (int, optional): Maximum number of batches to process (for debugging).

    Returns:
        float: The average RMSE on the validation set.
    """
    model.eval()
    running_rmse = 0.0
    total_samples = 0

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass: Model predicts the clean image
            clean_pred = model(inputs)

            # Reconstruct ground truth clean image: Clean = Input - Noise Target
            clean_target = inputs - targets

            # Metric: RMSE between cleaned pixel intensities and actual grayscale pixel intensities
            rmse = calculate_rmse(clean_pred, clean_target)

            batch_size = inputs.size(0)
            running_rmse += rmse.item() * batch_size
            total_samples += batch_size

    epoch_rmse = running_rmse / total_samples if total_samples > 0 else 0.0
    return epoch_rmse


def run_training(load_cached_data=True, max_epochs=None, max_batches_per_epoch=None):
    """
    Orchestrates the training pipeline, including data loading, model initialization,
    training loop, validation, early stopping, and checkpointing.

    Args:
        load_cached_data (bool): Whether to attempt loading cached data.
        max_epochs (int, optional): Override the number of epochs from Config.
        max_batches_per_epoch (int, optional): Limit batches per epoch for debugging.

    Returns:
        nn.Module: The model with the best validation weights loaded.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Load Data
    print("Loading Data...")
    train_loader, val_loader = get_dataloaders(load_cached_data=load_cached_data)

    # 2. Initialize Model
    print("Initializing Model...")
    model = SRDN().to(device)

    # 3. Setup Optimizer and Loss
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.MSELoss()

    # Scheduler: Reduce LR when validation metric stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # 4. Training Loop
    num_epochs = max_epochs if max_epochs is not None else Config.NUM_EPOCHS
    best_val_rmse = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training on {device} for {num_epochs} epochs...")

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            max_batches=max_batches_per_epoch,
        )

        # Validate
        val_rmse = validate(
            model, val_loader, device, max_batches=max_batches_per_epoch
        )

        # Step Scheduler
        scheduler.step(val_rmse)

        end_time = time.time()
        duration = end_time - start_time

        # Print Metrics (Full Precision)
        print(
            f"Epoch {epoch}/{num_epochs} | Time: {duration:.2f}s | Train Loss: {train_loss} | Val RMSE: {val_rmse}"
        )

        # Checkpoint and Early Stopping
        if val_rmse < best_val_rmse - Config.EARLY_STOPPING_MIN_DELTA:
            best_val_rmse = val_rmse
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_rmse, best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    print(f"Training complete. Best Validation RMSE: {best_val_rmse}")

    # Load best model weights
    if os.path.exists(best_model_path):
        print("Loading best model weights...")
        load_checkpoint(best_model_path, model, device=Config.DEVICE)

    return model
