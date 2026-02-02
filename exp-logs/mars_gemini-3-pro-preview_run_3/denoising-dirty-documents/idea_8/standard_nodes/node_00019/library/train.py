import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import set_seed
from library.model import ResDnCNN
from library.data_loader import get_dataloaders


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Runs one epoch of training.

    Args:
        model: The PyTorch model.
        dataloader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to run on.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        # loss.item() is the mean loss of the batch. Multiply by batch size to get total sum.
        running_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Runs validation on the dataset.

    Args:
        model: The PyTorch model.
        dataloader: Validation DataLoader.
        criterion: Loss function.
        device: Device to run on.

    Returns:
        tuple: (Average Loss, RMSE)
    """
    model.eval()
    running_loss = 0.0
    running_sse = 0.0  # Sum of Squared Errors for RMSE calculation
    total_samples = 0
    total_pixels = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            # Calculate SSE for RMSE (Sum of Squared Errors)
            # Since criterion is MSE, loss.item() * num_elements gives SSE
            # However, MSE is usually averaged over pixels then batch, or just all pixels.
            # PyTorch MSELoss default is 'mean'.
            # Total pixels in batch = batch_size * channels * height * width
            num_pixels_in_batch = inputs.numel()

            # Re-calculate raw SSE to be precise
            diff = outputs - targets
            sse = torch.sum(diff**2).item()

            running_sse += sse
            total_pixels += num_pixels_in_batch

    epoch_loss = running_loss / total_samples
    epoch_rmse = np.sqrt(running_sse / total_pixels)

    return epoch_loss, epoch_rmse


def train_model():
    """
    Main training function.
    Initializes model, data, optimizer, and runs the training loop with early stopping.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Training on device: {device}")

    # 2. Data
    print("Initializing DataLoaders...")
    train_loader, val_loader = get_dataloaders(load_cached_data=True)

    # 3. Model
    print("Initializing Model...")
    model = ResDnCNN().to(device)

    # 4. Optimization
    # Predicting noise residual, so we minimize MSE between predicted noise and true noise
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.COSINE_T_MAX, eta_min=Config.COSINE_ETA_MIN
    )

    # 5. Training Loop
    best_rmse = float("inf")
    early_stopping_counter = 0

    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")

    start_time = time.time()

    for epoch in range(Config.NUM_EPOCHS):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_rmse = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_duration = time.time() - epoch_start

        # Print Metrics (Full precision as requested)
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Time: {epoch_duration:.2f}s | "
            f"LR: {current_lr} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val RMSE: {val_rmse}"
        )

        # Checkpointing & Early Stopping
        if val_rmse < best_rmse:
            print(
                f"Validation RMSE improved from {best_rmse} to {val_rmse}. Saving model..."
            )
            best_rmse = val_rmse
            early_stopping_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        else:
            early_stopping_counter += 1
            print(
                f"No improvement. Early stopping counter: {early_stopping_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if early_stopping_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    total_time = time.time() - start_time
    print(f"Training complete in {total_time:.2f}s. Best Validation RMSE: {best_rmse}")
