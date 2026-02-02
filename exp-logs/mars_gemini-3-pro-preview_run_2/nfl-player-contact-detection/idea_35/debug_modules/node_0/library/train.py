import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, compute_mcc
from library.dataset import get_train_val_datasets
from library.model import APIRVNet
from library.loss import BinaryFocalLoss


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for inputs, targets in dataloader:
        # Unpack inputs (kinematic, visual) and move to device
        x_kin, x_vis = inputs
        x_kin = x_kin.to(device, non_blocking=True)
        x_vis = x_vis.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        # APIRVNet expects (x_kin, x_vis)
        logits = model(x_kin, x_vis)

        # Flatten logits to match target shape if necessary
        logits = logits.view(-1)

        # Calculate loss
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        running_loss += loss.item() * targets.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate_one_epoch(model, dataloader, criterion, device):
    """
    Performs validation on the validation set.
    Returns average loss and MCC score.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            x_kin, x_vis = inputs
            x_kin = x_kin.to(device, non_blocking=True)
            x_vis = x_vis.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(x_kin, x_vis)
            logits = logits.view(-1)

            loss = criterion(logits, targets)
            running_loss += loss.item() * targets.size(0)

            # For MCC calculation during training, we use a standard threshold of 0.0 (prob 0.5)
            # Threshold optimization happens post-training, but we need a proxy for Early Stopping.
            preds = (logits > 0.0).float()

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    epoch_loss = running_loss / len(dataloader.dataset)

    # Concatenate all predictions and targets
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    epoch_mcc = compute_mcc(all_targets, all_preds)

    return epoch_loss, epoch_mcc


def train_model(debug=False, max_epochs=None, batch_size=None):
    """
    Main function to train the APIRV-Net model.

    Args:
        debug (bool): If True, uses a subset of data for faster debugging.
        max_epochs (int): Override Config.MAX_EPOCHS if provided.
        batch_size (int): Override Config.BATCH_SIZE if provided.

    Returns:
        model (nn.Module): The trained model with best weights loaded.
        best_metric (float): The best validation MCC achieved.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    epochs = max_epochs if max_epochs is not None else Config.MAX_EPOCHS
    b_size = batch_size if batch_size is not None else Config.BATCH_SIZE

    print(f"Starting training on device: {device}")
    print(f"Configuration: Epochs={epochs}, Batch Size={b_size}, Debug={debug}")

    # 2. Data Loading
    print("Preparing datasets...")
    train_dataset, val_dataset = get_train_val_datasets(debug=debug)

    train_loader = DataLoader(
        train_dataset,
        batch_size=b_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch to maintain stable stats for BN
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=b_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    # Determine input dimensions dynamically based on dataset features
    kin_dim = len(train_dataset.kin_indices)
    vis_dim = len(train_dataset.vis_indices)

    print(f"Input Dimensions - Kinematic: {kin_dim}, Visual: {vis_dim}")

    model = APIRVNet(kin_input_dim=kin_dim, vis_input_dim=vis_dim)
    model.to(device)

    # 4. Optimization Setup
    criterion = BinaryFocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 5. Training Loop with Early Stopping
    best_mcc = -1.0
    patience = Config.EARLY_STOPPING_PATIENCE
    patience_counter = 0

    print("Starting training loop...")

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_mcc = validate_one_epoch(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{epochs} | Time: {elapsed:.2f}s | "
            f"Train Loss: {train_loss:.10f} | "
            f"Val Loss: {val_loss:.10f} | "
            f"Val MCC: {val_mcc:.10f}"
        )

        # Early Stopping Check
        if val_mcc > best_mcc:
            best_mcc = val_mcc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  -> New best model saved! (MCC: {best_mcc:.10f})")
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # 6. Load Best Weights
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print("Loading best model weights...")
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Warning: No model file saved. Returning current model.")

    return model, best_mcc
