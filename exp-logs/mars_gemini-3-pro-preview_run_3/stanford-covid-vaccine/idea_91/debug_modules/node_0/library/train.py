import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from library.config import Config
from library.utils import set_seed, mcrmse_loss, compute_score
from library.data import get_dataloaders
from library.model import RNAModel


def train_one_epoch(model, loader, optimizer, device, clip_grad):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in loader:
        features = batch["features"].to(device)
        pair_indices = batch["pair_indices"].to(device)
        pair_masks = batch["pair_masks"].to(device)
        targets = batch["targets"].to(device)

        batch_size = features.size(0)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(features, pair_indices, pair_masks)

        # Compute loss on all targets (Full Spectrum)
        loss = mcrmse_loss(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        # Accumulate loss (weighted by batch size for correct average)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using the competition metric.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            targets = batch["targets"]  # Keep on CPU

            outputs = model(features, pair_indices, pair_masks)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())

    # Concatenate all batches
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    # Compute score (handles slicing and column filtering)
    score = compute_score(y_pred, y_true)
    return score


def run_training(train_loader, val_loader):
    """
    Main training loop with Early Stopping and Best Model Saving.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Initialize Model
    model = RNAModel().to(device)

    # Optimizer (AdamW)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler (Cosine Annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # Training Loop
    best_score = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.MAX_EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, Config.GRAD_CLIP
        )
        val_score = validate(model, val_loader, device)

        # Step scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.MAX_EPOCHS} | Train Loss: {train_loss} | Val MCRMSE: {val_score}"
        )

        # Save Best Model
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            patience_counter = 0
            print(f"  New best model saved! Score: {best_score}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Validation Score: {best_score}")


def generate_submission(test_loader):
    """
    Generates predictions for the test set and saves to submission.csv.
    """
    device = torch.device(Config.DEVICE)

    # Load Best Model
    model = RNAModel().to(device)
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
        print("Loaded best model for inference.")
    else:
        print(
            "Warning: Best model not found. Using untrained model (should not happen in normal flow)."
        )

    model.eval()

    ids_list = []
    preds_list = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            ids = batch["id"]

            outputs = model(features, pair_indices, pair_masks)

            preds_list.append(outputs.cpu().numpy())
            ids_list.extend(ids)

    # Concatenate predictions: (N_test, 107, 5)
    all_preds = np.concatenate(preds_list, axis=0)

    # Prepare submission data
    # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    submission_rows = []
    target_cols = Config.TARGET_COLS

    for i, sample_id in enumerate(ids_list):
        sample_pred = all_preds[i]  # Shape (107, 5)

        for seqpos in range(Config.SEQ_LENGTH):
            row_id = f"{sample_id}_{seqpos}"
            # Get the 5 predicted values
            vals = sample_pred[seqpos].tolist()

            # Create row
            row = [row_id] + vals
            submission_rows.append(row)

    # Create DataFrame
    columns = ["id_seqpos"] + target_cols
    submission_df = pd.DataFrame(submission_rows, columns=columns)

    # Save
    submission_df.to_csv(Config.SUBMISSION_FILE_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE_PATH}")


def main():
    """
    Orchestrates the training and submission pipeline.
    """
    # Load Data
    train_loader, val_loader, test_loader = get_dataloaders()

    # Run Training
    run_training(train_loader, val_loader)

    # Generate Submission
    generate_submission(test_loader)
