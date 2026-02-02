import os
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from library.config import Config
from library.utils import seed_everything, get_device
from library.data import get_dataloaders
from library.model import CalibratedHierarchicalSeqModel
from library.loss import WeightedMultiLabelLoss


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    accumulation_steps: int,
    max_grad_norm: float,
) -> float:
    """
    Executes one training epoch with gradient accumulation.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    optimizer.zero_grad()

    for batch_idx, (images, targets) in enumerate(loader):
        batch_size = images.size(0)

        # Move data to device
        images = images.to(device, dtype=torch.float32)
        targets = targets.to(device, dtype=torch.float32)

        # Forward pass
        logits = model(images)

        # Calculate loss
        loss = loss_fn(logits, targets)

        # Normalize loss for gradient accumulation
        loss_normalized = loss / accumulation_steps

        # Backward pass
        loss_normalized.backward()

        # Update weights if accumulation steps reached
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        # Accumulate metrics
        # We multiply by batch_size because loss_fn returns the mean loss over the batch
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    # Handle remaining gradients if total batches not divisible by accumulation_steps
    if len(loader) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    with torch.no_grad():
        for images, targets in loader:
            batch_size = images.size(0)
            images = images.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)

            logits = model(images)
            loss = loss_fn(logits, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

    val_loss = running_loss / dataset_size
    return val_loss


def run_training(
    epochs: int = Config.EPOCHS,
    batch_size: int = Config.BATCH_SIZE,
    accumulation_steps: int = Config.ACCUMULATION_STEPS,
    learning_rate: float = Config.LEARNING_RATE,
    weight_decay: float = Config.WEIGHT_DECAY,
    patience: int = Config.PATIENCE,
    debug: bool = Config.DEBUG,
):
    """
    Main orchestration function for training.
    """
    # Override Config with runtime arguments
    Config.EPOCHS = epochs
    Config.BATCH_SIZE = batch_size
    Config.ACCUMULATION_STEPS = accumulation_steps
    Config.LEARNING_RATE = learning_rate
    Config.WEIGHT_DECAY = weight_decay
    Config.PATIENCE = patience
    Config.DEBUG = debug

    # Setup
    seed_everything(Config.SEED)
    device = get_device()

    # Ensure output directories
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    print(f"Initializing training on {device}...")
    print(
        f"Configuration: Epochs={epochs}, Batch={batch_size}, Accum={accumulation_steps}, LR={learning_rate}, Debug={debug}"
    )

    # Data Loading
    # We use load_cached_data=True to utilize the parquet cache if available
    train_loader, val_loader = get_dataloaders(load_cached_data=True)

    # Model Initialization
    model = CalibratedHierarchicalSeqModel(pretrained=True)
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Loss Function
    loss_fn = WeightedMultiLabelLoss().to(device)

    # Training Loop State
    best_val_loss = float("inf")
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            accumulation_steps=accumulation_steps,
            max_grad_norm=Config.MAX_GRAD_NORM,
        )

        # Validate
        val_loss = validate(
            model=model, loader=val_loader, loss_fn=loss_fn, device=device
        )

        epoch_duration = time.time() - epoch_start

        # Logging
        print(f"Epoch {epoch}/{epochs} | Time: {epoch_duration:.2f}s")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")

        # Checkpointing & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New best model saved to {Config.BEST_MODEL_PATH}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        # Save latest checkpoint
        torch.save(model.state_dict(), Config.LAST_MODEL_PATH)

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    total_time = time.time() - start_time
    print(f"Training complete. Total time: {total_time:.2f}s")
    print(f"Best Validation Loss: {best_val_loss}")
