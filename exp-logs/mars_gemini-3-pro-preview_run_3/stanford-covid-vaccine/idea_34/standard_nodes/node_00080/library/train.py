import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from library.config import config
from library.utils import set_seed, calculate_mcrmse, format_submission
from library.data import get_dataloaders
from library.model import DCASGBiGRU


class MCRMSELoss(nn.Module):
    """
    Column-wise Root Mean Squared Error Loss.
    Calculates RMSE for each column independently and returns the mean.
    """

    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        # preds, targets shape: (Batch, Seq, Cols)
        # Calculate MSE per column (averaging over Batch and Seq dims)
        mse = torch.mean((preds - targets) ** 2, dim=(0, 1))
        # RMSE per column (add epsilon for stability)
        rmse = torch.sqrt(mse + 1e-8)
        # Mean of RMSEs across the 5 targets
        return torch.mean(rmse)


def train_one_epoch(model, loader, optimizer, criterion, device, max_grad_norm):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        inputs = batch["inputs"].to(device)
        pair_indices = batch["pair_indices"].to(device)
        pair_masks = batch["pair_masks"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = model(inputs, pair_indices, pair_masks)

        # Slice to scored sequence length for loss calculation
        # Ground truth is only valid for the first 68 positions (SEQ_SCORED)
        preds_sliced = preds[:, : config.SEQ_SCORED, :]
        targets_sliced = targets[:, : config.SEQ_SCORED, :]

        # Compute loss on all 5 targets
        loss = criterion(preds_sliced, targets_sliced)

        # Backward pass
        loss.backward()

        # Gradient Clipping (Mandatory for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using the official metric logic.
    Aggregates all predictions first to ensure global metric calculation.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            targets = batch["targets"]  # Keep on CPU

            preds = model(inputs, pair_indices, pair_masks)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())

    # Concatenate to form full dataset arrays
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate MCRMSE using utility
    # This utility handles slicing to SEQ_SCORED and filtering for the 3 scored columns
    score = calculate_mcrmse(all_preds, all_targets)

    return score


def generate_submission(test_loader, device):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("Generating submission...")

    # Load best model
    model = DCASGBiGRU().to(device)
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            ids = batch["id"]

            preds = model(inputs, pair_indices, pair_masks)

            all_preds.append(preds.cpu().numpy())
            all_ids.extend(ids)

    all_preds = np.concatenate(all_preds, axis=0)

    # Format submission
    # preds must be (N, 107, 5) - format_submission handles flattening
    submission_df = format_submission(all_ids, all_preds)

    # Save
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
    # Print first few rows for verification
    print(submission_df.head())


def run_training(debug=False):
    """
    Main orchestration function for training and inference.
    """
    # Setup
    set_seed(config.SEED)
    device = torch.device(config.DEVICE)

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(debug=debug)

    # Model
    model = DCASGBiGRU().to(device)

    # Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Scheduler (Cosine Annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS, eta_min=config.ETA_MIN
    )

    criterion = MCRMSELoss()

    # Training Loop
    best_score = float("inf")
    patience_counter = 0

    print(f"Starting training on {device} for {config.EPOCHS} epochs...")

    for epoch in range(config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, config.MAX_GRAD_NORM
        )

        # Validate
        val_score = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        elapsed = time.time() - start_time

        # Print metrics (Full precision for validation score)
        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCRMSE: {val_score} | "
            f"Time: {elapsed:.2f}s"
        )

        # Checkpointing & Early Stopping
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"  New best model saved! Score: {best_score}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{config.PATIENCE}")

        if patience_counter >= config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val MCRMSE: {best_score}")

    # Generate Final Submission
    generate_submission(test_loader, device)
