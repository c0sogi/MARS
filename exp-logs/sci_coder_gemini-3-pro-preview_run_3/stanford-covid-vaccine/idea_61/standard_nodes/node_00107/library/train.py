import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import seed_everything, MCRMSELoss
from library.data import get_dataloaders
from library.model import RNAModel


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Move inputs to device
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device)

        # Structural interaction data
        pair_index = None
        pair_mask = None
        if "pair_index" in batch:
            pair_index = batch["pair_index"].to(device)
            pair_mask = batch["pair_mask"].to(device)

        optimizer.zero_grad()

        # Forward pass
        # Inputs: (B, 107, 14) -> Outputs: (B, 107, 5)
        outputs = model(inputs, pair_index=pair_index, pair_mask=pair_mask)

        # Calculate loss on ALL 5 targets (Multi-Task Learning)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping (Critical for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Calculates MCRMSE only on the 3 scored columns for the first 68 positions.
    """
    model.eval()

    # Indices for the scored columns: reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (3)
    # Config.TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    scored_indices = [0, 1, 3]

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)

            pair_index = None
            pair_mask = None
            if "pair_index" in batch:
                pair_index = batch["pair_index"].to(device)
                pair_mask = batch["pair_mask"].to(device)

            outputs = model(inputs, pair_index=pair_index, pair_mask=pair_mask)

            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())

    # Concatenate all batches
    preds = torch.cat(all_preds, dim=0)  # (N, 107, 5)
    targets = torch.cat(all_targets, dim=0)  # (N, 107, 5)

    # 1. Slice to scored sequence length (first 68 positions)
    preds_sliced = preds[:, : Config.PRED_LEN, :]
    targets_sliced = targets[:, : Config.PRED_LEN, :]

    # 2. Filter for scored columns only
    preds_scored = preds_sliced[:, :, scored_indices]
    targets_scored = targets_sliced[:, :, scored_indices]

    # Calculate MCRMSE
    score = criterion(preds_scored, targets_scored).item()

    return score


def run_training(epochs=Config.EPOCHS, load_cached_data=True):
    """
    Main training loop with Early Stopping.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Data
    print("Initializing DataLoaders...")
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=load_cached_data)

    # 3. Model
    print("Initializing Model...")
    model = RNAModel().to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    criterion = MCRMSELoss()

    # 5. Training Loop
    best_score = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_score = validate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step()

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss (All 5): {train_loss:.6f} | "
            f"Val MCRMSE (Scored 3): {val_score} | "
            f"Time: {elapsed:.2f}s"
        )

        # Checkpoint & Early Stopping
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  -> New Best Model Saved! Score: {best_score}")
        else:
            patience_counter += 1
            print(
                f"  -> No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Score: {best_score}")
