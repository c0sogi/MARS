import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import Config
from library.utils import set_seed, get_device, calculate_mcrmse
from library.loss import MCRMSELoss
from library.data import get_loaders
from library.model import RNAModel


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Move data to device
        inputs = batch["inputs"].to(device)
        pair_indices = batch["pair_indices"].to(device)
        pair_masks = batch["pair_masks"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = model(inputs, pair_indices, pair_masks)

        # Compute loss
        loss = criterion(preds, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD_NORM)

        # Optimizer step
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def evaluate(model, loader, device):
    """
    Evaluates the model on the validation set using the competition metric.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            targets = batch[
                "targets"
            ]  # Keep targets on CPU for metric calc if needed, or move later

            preds = model(inputs, pair_indices, pair_masks)

            all_preds.append(preds.detach().cpu())
            all_targets.append(targets)

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate MCRMSE
    score = calculate_mcrmse(all_preds, all_targets)

    return score


def generate_submission(model, loader, device):
    """
    Generates predictions for the test set and creates the submission CSV.
    """
    model.eval()
    submission_data = []

    # Columns required in submission
    target_cols = Config.TARGET_COLS

    print("Generating predictions for submission...")

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            ids = batch["ids"]  # List of strings

            preds = model(inputs, pair_indices, pair_masks)
            preds = preds.detach().cpu().numpy()  # (B, 107, 5)

            # Unpack batch
            batch_size = preds.shape[0]
            seq_len = preds.shape[1]

            for b in range(batch_size):
                sample_id = ids[b]
                sample_preds = preds[b]  # (107, 5)

                for s in range(seq_len):
                    # Construct row dictionary
                    row_id = f"{sample_id}_{s}"
                    row_data = {"id_seqpos": row_id}

                    # Add predictions
                    for i, col in enumerate(target_cols):
                        row_data[col] = float(sample_preds[s, i])

                    submission_data.append(row_data)

    # Create DataFrame
    df_sub = pd.DataFrame(submission_data)

    # Save to CSV
    save_path = Config.SUBMISSION_PATH
    df_sub.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")


def run_training():
    """
    Main function to run the training pipeline.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = get_device()
    print(f"Using device: {device}")

    # 2. Data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # 3. Model
    print("Initializing model...")
    model = RNAModel(config=Config).to(device)

    # 4. Optimizer, Scheduler, Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    criterion = MCRMSELoss()

    # 5. Training Loop
    best_score = float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_score = evaluate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCRMSE: {val_score} | "  # Printing full precision
            f"LR: {current_lr:.2e}"
        )

        # Early Stopping & Checkpointing
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  New best model saved! Score: {best_score}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 6. Submission
    print("Loading best model for submission...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    generate_submission(model, test_loader, device)
