import os
import torch
import numpy as np
from library.config import (
    MODEL_SAVE_PATH,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    PATIENCE,
    SCORED_INDICES,
)
from library.data import get_dataloaders
from library.model import LFDCN
from library.utils import set_seed, mcrmse_loss


def train_one_epoch(model, loader, optimizer, device):
    """
    Trains the model for one epoch using the iterative refinement loss.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        inputs, partner_indices, targets, masks = batch
        inputs = inputs.to(device)
        partner_indices = partner_indices.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # LFDCN forward pass with targets returns the weighted combined loss:
        # Loss = MCRMSE(pred_2) + 0.5 * MCRMSE(pred_1)
        loss, _ = model(inputs, partner_indices, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Computes the global MCRMSE on the final predictions (Pass 2) for scored columns.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            inputs, partner_indices, targets, masks = batch
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)
            targets = targets.to(device)

            # Forward pass in inference mode (targets=None) returns the final prediction (pred_2)
            preds = model(inputs, partner_indices, targets=None)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    # Concatenate all batches to compute global metric
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate Global MCRMSE on scored columns
    # mcrmse_loss handles the slicing using SCORED_INDICES by default
    score = mcrmse_loss(all_preds, all_targets)

    return score.item()


def train_model():
    """
    Main training routine handling initialization, loops, and early stopping.
    """
    # Set reproducibility
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # Load Data
    # Caching is handled internally by get_dataloaders -> load_and_cache_data
    train_loader, val_loader, _ = get_dataloaders(
        load_cached_data=True, batch_size=BATCH_SIZE
    )

    # Initialize Model
    model = LFDCN().to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    best_val_score = float("inf")
    early_stop_count = 0

    print("Starting Training...")

    for epoch in range(EPOCHS):
        # Training Step
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validation Step
        val_score = validate(model, val_loader, device)

        # Print metrics (Full precision for validation score as requested)
        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score}"
        )

        # Scheduler Step
        scheduler.step(val_score)

        # Checkpointing & Early Stopping
        if val_score < best_val_score:
            best_val_score = val_score
            early_stop_count = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"New Best Model Saved! Score: {best_val_score}")
        else:
            early_stop_count += 1
            if early_stop_count >= PATIENCE:
                print("Early stopping triggered.")
                break
