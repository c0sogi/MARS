import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os

from library.config import Config
from library.utils import mcrmse_numpy, seed_everything
from library.loss import MCRMSELoss
from library.model import SDBR_BiGRU
from library.data import get_dataloaders


def train_one_epoch(model, dataloader, optimizer, criterion, device, max_grad_norm):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        inputs = batch["inputs"].to(device)
        pair_index = batch["pair_index"].to(device)
        pair_mask = batch["pair_mask"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = model(inputs, pair_index, pair_mask)

        # Slice predictions to match targets (seq_scored=68)
        # Targets are (B, 68, 5), Preds are (B, 107, 5)
        preds_sliced = preds[:, : Config.SEQ_SCORED, :]

        # Compute loss on all 5 columns
        loss = criterion(preds_sliced, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Optimizer step
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, device):
    """
    Performs validation and calculates MCRMSE on scored columns.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["inputs"].to(device)
            pair_index = batch["pair_index"].to(device)
            pair_mask = batch["pair_mask"].to(device)
            targets = batch["targets"].numpy()  # Keep on CPU for metric calculation

            preds = model(inputs, pair_index, pair_mask)
            preds = preds.cpu().numpy()

            all_preds.append(preds)
            all_targets.append(targets)

    # Concatenate all batches
    y_pred = np.concatenate(all_preds, axis=0)  # (N, 107, 5)
    y_true = np.concatenate(all_targets, axis=0)  # (N, 68, 5)

    # Calculate metric
    # mcrmse_numpy handles slicing y_pred to 68 and selecting the 3 scored columns
    score = mcrmse_numpy(y_true, y_pred)

    return score


def train_pipeline(epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, debug=False):
    """
    Main training pipeline with Early Stopping.
    """
    seed_everything(Config.SEED)
    Config.setup_directories()
    device = torch.device(Config.DEVICE)

    # Load Data
    train_loader, val_loader, _ = get_dataloaders(
        load_cached_data=True, batch_size=batch_size
    )

    if debug:
        epochs = 2
        print("Debug mode: Running for 2 epochs.")

    # Initialize Model
    model = SDBR_BiGRU().to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.T_MAX)

    # Loss Function
    criterion = MCRMSELoss()

    best_score = float("inf")
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, Config.MAX_GRAD_NORM
        )

        # Update Scheduler
        scheduler.step()

        # Validate
        val_score = validate(model, val_loader, device)

        # Print metrics (Full precision)
        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Val MCRMSE: {val_score}"
        )

        # Early Stopping & Checkpointing
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Validation Score: {best_score}")


def predict_and_submit(batch_size=Config.BATCH_SIZE):
    """
    Generates predictions for the test set and saves the submission file.
    """
    seed_everything(Config.SEED)
    Config.setup_directories()
    device = torch.device(Config.DEVICE)

    # Load Test Data
    _, _, test_loader = get_dataloaders(load_cached_data=True, batch_size=batch_size)

    # Load Model
    model = SDBR_BiGRU().to(device)
    if not os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Error: Model checkpoint not found at {Config.BEST_MODEL_PATH}")
        return

    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    all_preds = []
    all_ids = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"].to(device)
            pair_index = batch["pair_index"].to(device)
            pair_mask = batch["pair_mask"].to(device)
            ids = batch["id"]

            preds = model(inputs, pair_index, pair_mask)
            preds = preds.cpu().numpy()

            all_preds.append(preds)
            all_ids.extend(ids)

    # Concatenate predictions
    all_preds = np.concatenate(all_preds, axis=0)  # (N, 107, 5)

    # Prepare submission rows
    submission_rows = []

    for i, sample_id in enumerate(all_ids):
        sample_preds = all_preds[i]
        for seqpos in range(Config.SEQ_LENGTH):
            row_id = f"{sample_id}_{seqpos}"
            row_data = {
                "id_seqpos": row_id,
                "reactivity": sample_preds[seqpos, 0],
                "deg_Mg_pH10": sample_preds[seqpos, 1],
                "deg_pH10": sample_preds[seqpos, 2],
                "deg_Mg_50C": sample_preds[seqpos, 3],
                "deg_50C": sample_preds[seqpos, 4],
            }
            submission_rows.append(row_data)

    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
