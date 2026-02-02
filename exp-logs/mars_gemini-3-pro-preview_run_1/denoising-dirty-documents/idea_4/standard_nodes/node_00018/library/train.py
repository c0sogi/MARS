import os
import time
import torch
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import seed_everything, calculate_rmse
from library.dataset import get_dataloaders
from library.model import DeepSupervisionUNet
import torch.nn as nn


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Handles the training loop for a single epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (noisy, clean) in enumerate(loader):
        noisy = noisy.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()

        # Forward pass
        # If deep supervision is on and model is in train mode, returns a list of tensors
        outputs = model(noisy)

        # Compute loss
        # MultiScaleMSELoss handles list of outputs automatically
        loss = criterion(outputs, clean)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using RMSE.
    """
    model.eval()
    total_rmse = 0.0
    num_samples = 0

    with torch.no_grad():
        for noisy, clean in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            # Forward pass
            # In eval mode, model returns only the final prediction tensor
            prediction = model(noisy)

            # Clamp predictions to valid pixel range [0, 1]
            prediction = torch.clamp(prediction, 0.0, 1.0)

            # Calculate RMSE for this image
            # Validation batch size is 1, so we calculate per image
            rmse = calculate_rmse(prediction, clean)
            total_rmse += rmse
            num_samples += 1

    return total_rmse / num_samples if num_samples > 0 else 0.0


def train_model(model_index=0, load_cached_data=True):
    """
    Main function to train a single model instance.

    Args:
        model_index (int): Index of the model in the ensemble (used for saving).
        load_cached_data (bool): Whether to load data from cache.
    """
    # 1. Setup
    seed_everything(Config.SEED + model_index)
    device = torch.device(Config.DEVICE)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"--- Starting training for Model {model_index} ---")
    print(f"Device: {device}")

    # 2. Data
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=load_cached_data)

    # 3. Model
    model = DeepSupervisionUNet().to(device)

    # 4. Optimization
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=0
    )

    # Loss Function
    criterion = nn.MSELoss().to(device)

    # 5. Training Loop
    best_rmse = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, f"model_{model_index}.pth")

    start_time = time.time()

    for epoch in range(Config.NUM_EPOCHS):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_rmse = validate(model, val_loader, device)

        # Step Scheduler
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # Checkpointing
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(model.state_dict(), best_model_path)
            saved_str = " [Saved Best]"
        else:
            saved_str = ""

        # Logging
        # Printing full precision as requested
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.10f} | "
            f"Val RMSE: {val_rmse:.10f}{saved_str}"
        )

    total_time = time.time() - start_time
    print(
        f"Training finished for Model {model_index}. "
        f"Best Val RMSE: {best_rmse:.10f}. "
        f"Total time: {total_time:.2f}s"
    )

    return best_rmse
