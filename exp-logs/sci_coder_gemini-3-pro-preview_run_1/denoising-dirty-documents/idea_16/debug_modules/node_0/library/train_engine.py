import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import (
    WORKING_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    OPTIMIZER_NAME,
    SCHEDULER_NAME,
    SCHEDULER_T_MAX,
    NUM_WORKERS,
    DEVICE,
)
from library.utils import set_seed, get_device
from library.model import ResolutionPreservedUNet
from library.dataset import DenoisingDataset


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network.
        dataloader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        device (torch.device): Device to run on.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for noisy_imgs, clean_imgs, _ in dataloader:
        noisy_imgs = noisy_imgs.to(device)
        clean_imgs = clean_imgs.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(noisy_imgs)

        # Compute loss
        loss = criterion(outputs, clean_imgs)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / num_batches if num_batches > 0 else 0.0


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network.
        dataloader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run on.

    Returns:
        float: Average RMSE over the validation set.
    """
    model.eval()
    running_mse = 0.0
    num_batches = 0

    with torch.no_grad():
        for noisy_imgs, clean_imgs, _ in dataloader:
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)

            outputs = model(noisy_imgs)

            # Criterion is MSE, so loss.item() is the Mean Squared Error for this batch
            loss = criterion(outputs, clean_imgs)

            running_mse += loss.item()
            num_batches += 1

    avg_mse = running_mse / num_batches if num_batches > 0 else 0.0
    rmse = np.sqrt(avg_mse)
    return rmse


def train_model(stream_config, seed_index, debug_max_samples=None):
    """
    Main training loop for a specific stream and seed.

    Args:
        stream_config (dict): Configuration dict containing 'name', 'img_size', 'seeds'.
        seed_index (int): Index of the seed to use from stream_config['seeds'].
        debug_max_samples (int, optional): Limit dataset size for debugging.
    """
    # 1. Setup
    stream_name = stream_config["name"]
    seed = stream_config["seeds"][seed_index]
    img_size = stream_config["img_size"]

    set_seed(seed)
    device = get_device()
    print(f"Starting training for {stream_name} | Seed: {seed} | Device: {device}")

    # 2. Data Loading
    # Load metadata
    train_df = pd.read_csv(TRAIN_METADATA_PATH)
    val_df = pd.read_csv(VAL_METADATA_PATH)

    # Debugging: Limit data if requested
    if debug_max_samples is not None:
        print(f"DEBUG MODE: Limiting training data to {debug_max_samples} samples.")
        train_df = train_df.head(debug_max_samples)
        val_df = val_df.head(debug_max_samples)

    # Initialize Datasets
    # Train: Augmentation + Random Crops (Stream specific size)
    train_dataset = DenoisingDataset(
        train_df,
        img_size=img_size,
        augment=True,
        cache_name=f"train_cache",
        load_cached_data=True,
    )

    # Val: No Augmentation + Padding (Inference mode)
    val_dataset = DenoisingDataset(
        val_df,
        img_size=None,
        augment=False,
        cache_name=f"val_cache",
        load_cached_data=True,
    )

    # Initialize Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Validation images vary in size after padding, so batch_size=1 is safest
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model & Optimization
    model = ResolutionPreservedUNet().to(device)

    # Optimizer
    if OPTIMIZER_NAME == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    else:
        # Fallback or extension point
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Scheduler
    if SCHEDULER_NAME == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=SCHEDULER_T_MAX
        )
    else:
        scheduler = None

    # Loss Function (MSE)
    criterion = nn.MSELoss()

    # 4. Training Loop
    best_rmse = float("inf")
    model_save_path = os.path.join(WORKING_DIR, f"{stream_name}_seed_{seed}.pth")

    print(f"Training for {EPOCHS} epochs...")

    for epoch in range(1, EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_rmse = validate(model, val_loader, criterion, device)

        # Step Scheduler
        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = LEARNING_RATE

        # Checkpoint
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(model.state_dict(), model_save_path)
            print(
                f"Epoch {epoch}: New Best RMSE: {best_rmse} (Saved to {os.path.basename(model_save_path)})"
            )

        # Logging (Every 10 epochs or first/last)
        if epoch % 10 == 0 or epoch == 1 or epoch == EPOCHS:
            print(
                f"Epoch {epoch}/{EPOCHS} | Train Loss (MSE): {train_loss} | Val RMSE: {val_rmse} | LR: {current_lr}"
            )

    print(f"Training complete. Best Validation RMSE: {best_rmse}")
    print(f"Model saved to: {model_save_path}")
    return best_rmse
