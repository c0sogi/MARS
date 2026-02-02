import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import set_seed, calculate_mcrmse
from library.data import get_dataloaders
from library.model import VectorScaledWideStreamBiGRU


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Move data to device
        seq = batch["seq"].to(device)
        loop = batch["loop"].to(device)
        pair_enc = batch["pair_enc"].to(device)
        targets = batch["target"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        preds = model(seq, loop, pair_enc)

        # Loss Calculation:
        # We only calculate loss on the scored positions (first 68).
        # Slicing is equivalent to masking here since the mask is fixed for the first 68.
        preds_scored = preds[:, : Config.SCORED_LENGTH, :]
        targets_scored = targets[:, : Config.SCORED_LENGTH, :]

        loss = criterion(preds_scored, targets_scored)

        # Backward pass
        loss.backward()

        # Gradient Clipping (Critical for Width 512 stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP_NORM)

        # Optimizer step
        optimizer.step()

        running_loss += loss.item() * seq.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using MCRMSE.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            pair_enc = batch["pair_enc"].to(device)
            targets = batch["target"].to(device)

            preds = model(seq, loop, pair_enc)

            # Slice to scored length for metric calculation
            preds_scored = preds[:, : Config.SCORED_LENGTH, :]
            targets_scored = targets[:, : Config.SCORED_LENGTH, :]

            all_preds.append(preds_scored.cpu().numpy())
            all_targets.append(targets_scored.cpu().numpy())

    # Concatenate all batches
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    # Calculate MCRMSE
    score = calculate_mcrmse(y_true, y_pred)
    return score


def run_training():
    """
    Main function to run the training pipeline.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data
    print("Loading data...")
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=True)

    # 3. Model
    print("Initializing model...")
    model = VectorScaledWideStreamBiGRU().to(device)

    # 4. Optimizer & Scheduler
    # Low weight decay to preserve recurrent signals
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # Loss Function (MSE)
    criterion = nn.MSELoss()

    # 5. Training Loop
    best_mcrmse = float("inf")

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_mcrmse = validate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()

        # Logging
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCRMSE: {val_mcrmse:.10f}"
        )

        # Checkpoint
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            print(f"New best model found! Saving to {Config.MODEL_SAVE_PATH}")
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    print(f"Training complete. Best Val MCRMSE: {best_mcrmse:.10f}")


if __name__ == "__main__":
    run_training()
