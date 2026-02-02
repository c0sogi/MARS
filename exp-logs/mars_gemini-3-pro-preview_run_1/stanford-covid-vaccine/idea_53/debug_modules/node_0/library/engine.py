import torch
import torch.optim as optim
import numpy as np
import os
import sys

from library.config import Config
from library.utils import get_device, seed_all, mcrmse
from library.dataset import get_dataloader
from library.model import RNAModel, loss_fn, generate_submission


def train_fn(
    model, dataloader, optimizer, device, clip_grad=Config.CLIP_GRAD, max_batches=None
):
    """
    Performs one epoch of training.

    Args:
        model: The PyTorch model.
        dataloader: Training dataloader.
        optimizer: Optimizer instance.
        device: Torch device.
        clip_grad: Gradient clipping threshold.
        max_batches: Optional integer to limit the number of batches (for debugging).

    Returns:
        float: Average training loss.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break

        # Move inputs to device
        seq = batch["sequence"].to(device)
        loop = batch["loop_type"].to(device)
        dist = batch["pairing_distance"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()

        # Forward pass
        pred = model(seq, loop, dist)

        # Calculate loss (Masked MSE)
        loss = loss_fn(pred, target)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def eval_fn(model, dataloader, device, max_batches=None):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: Validation dataloader.
        device: Torch device.
        max_batches: Optional integer to limit the number of batches.

    Returns:
        float: MCRMSE score.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break

            seq = batch["sequence"].to(device)
            loop = batch["loop_type"].to(device)
            dist = batch["pairing_distance"].to(device)
            target = batch["target"].to(device)

            pred = model(seq, loop, dist)

            # Slice to scored region for metric calculation
            # target shape: (B, 68, 3)
            # pred shape: (B, 107, 3) -> slice to (B, 68, 3)
            seq_scored = target.shape[1]
            pred_scored = pred[:, :seq_scored, :]

            all_preds.append(pred_scored.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    if len(all_preds) == 0:
        return 0.0

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate MCRMSE
    score = mcrmse(all_targets, all_preds)
    return score


def run_training(config=Config, epochs=None, patience=5, debug=False):
    """
    Orchestrates the training pipeline with Early Stopping and Submission Generation.

    Args:
        config: Configuration class.
        epochs: Number of epochs to train (overrides config if provided).
        patience: Number of epochs to wait for improvement before early stopping.
        debug: If True, runs for a limited number of batches per epoch.
    """
    # Setup
    seed_all(config.SEED)
    device = get_device()
    print(f"Using device: {device}")

    train_epochs = epochs if epochs is not None else config.EPOCHS
    max_batches = 10 if debug else None

    # Data Loaders
    print("Initializing DataLoaders...")
    train_loader = get_dataloader("train", batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = get_dataloader("val", batch_size=config.BATCH_SIZE, shuffle=False)

    # Model
    print("Initializing Model...")
    model = RNAModel(config).to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_epochs)

    # Training Loop
    best_score = float("inf")
    patience_counter = 0

    print(f"Starting training for {train_epochs} epochs (Patience: {patience})...")

    for epoch in range(train_epochs):
        train_loss = train_fn(
            model,
            train_loader,
            optimizer,
            device,
            clip_grad=config.CLIP_GRAD,
            max_batches=max_batches,
        )

        val_score = eval_fn(model, val_loader, device, max_batches=max_batches)

        scheduler.step()

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{train_epochs} | Train Loss: {train_loss} | Val MCRMSE: {val_score}"
        )

        # Save best model
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), config.MODEL_PATH)
            # print(f"New best model saved with MCRMSE: {best_score}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training complete. Best Val MCRMSE: {best_score}")

    # Generate Submission
    # This function loads the best model from config.MODEL_PATH
    generate_submission(model, config, device)
