import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import seed_everything, MCRMSELoss
from library.data import get_dataset, RNADataset
from library.model import HighCapacityBiGRU


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Unpack batch
        item, y = batch
        x = item["X"].to(device)
        adj = item["adj"].to(device)
        mask = item["mask"].to(device)
        y = y.to(device)  # (B, 68, 5)

        optimizer.zero_grad()

        # Forward pass (B, 107, 5)
        preds = model(x, adj, mask)

        # Slice to scored sequence length (68)
        preds_sliced = preds[:, : Config.SEQ_SCORED, :]

        # Compute loss on all 5 targets
        loss = criterion(preds_sliced, y)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Computes MCRMSE only on the 3 scored columns: reactivity, deg_Mg_pH10, deg_Mg_50C.
    """
    model.eval()
    all_preds = []
    all_targets = []

    # Scored column indices: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_indices = [0, 1, 3]

    with torch.no_grad():
        for batch in loader:
            item, y = batch
            x = item["X"].to(device)
            adj = item["adj"].to(device)
            mask = item["mask"].to(device)

            # Forward pass
            preds = model(x, adj, mask)

            # Slice to scored length
            preds_sliced = preds[:, : Config.SEQ_SCORED, :]

            all_preds.append(preds_sliced.cpu())
            all_targets.append(y.cpu())

    # Concatenate
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Select only the scored columns for metric calculation
    # Shape: (N, 68, 3)
    preds_scored = all_preds[:, :, scored_indices]
    targets_scored = all_targets[:, :, scored_indices]

    # Compute MCRMSE manually for validation to ensure correct column averaging
    preds_flat = preds_scored.reshape(-1, preds_scored.shape[-1])
    targets_flat = targets_scored.reshape(-1, targets_scored.shape[-1])

    mse = torch.mean((preds_flat - targets_flat) ** 2, dim=0)
    rmse = torch.sqrt(mse)  # RMSE per column
    mcrmse = torch.mean(rmse).item()  # Mean of column RMSEs

    return mcrmse


def train_model(debug=False, epochs=None, patience=15):
    """
    Main training function.

    Args:
        debug (bool): If True, limits dataset size for quick debugging.
        epochs (int): Override number of epochs from Config.
        patience (int): Early stopping patience.
    """
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Load Data
    train_inputs, train_targets = get_dataset("train", load_cached_data=True)
    val_inputs, val_targets = get_dataset("val", load_cached_data=True)

    if debug:
        print("Debug mode: slicing datasets to 100 samples.")
        for k in train_inputs:
            train_inputs[k] = train_inputs[k][:100]
        train_targets = train_targets[:100]
        for k in val_inputs:
            val_inputs[k] = val_inputs[k][:100]
        val_targets = val_targets[:100]

    train_dataset = RNADataset(train_inputs, train_targets)
    val_dataset = RNADataset(val_inputs, val_targets)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Initialize Model
    model = HighCapacityBiGRU(Config).to(device)

    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    num_epochs = epochs if epochs is not None else Config.EPOCHS
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Loss function
    criterion = MCRMSELoss()

    # Training Loop
    best_score = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    early_stop_counter = 0

    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_score = validate(model, val_loader, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss} | Val MCRMSE: {val_score}"
        )

        # Checkpointing
        if val_score < best_score:
            best_score = val_score
            early_stop_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with score: {best_score}")
        else:
            early_stop_counter += 1

        # Early Stopping
        if early_stop_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Validation Score: {best_score}")
    return best_score


def generate_submission():
    """
    Generates predictions for the test set and saves the submission file.
    """
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        print("Error: Best model not found. Cannot generate submission.")
        return

    # Load Test Data
    test_inputs = get_dataset("test", load_cached_data=True)
    test_dataset = RNADataset(test_inputs, targets=None)
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    # Load Model
    model = HighCapacityBiGRU(Config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("Generating predictions...")
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            item = batch
            x = item["X"].to(device)
            adj = item["adj"].to(device)
            mask = item["mask"].to(device)

            # Predict for full length (107)
            preds = model(x, adj, mask)  # (B, 107, 5)
            all_preds.append(preds.cpu().numpy())

    # Concatenate all predictions: (N_test, 107, 5)
    all_preds = np.concatenate(all_preds, axis=0)

    # Prepare Submission DataFrame
    ids = test_inputs["ids"]
    submission_rows = []

    # Iterate through samples and sequence positions
    for i, sample_id in enumerate(ids):
        sample_preds = all_preds[i]  # (107, 5)
        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            # Get the 5 predicted values
            vals = sample_preds[seqpos].tolist()
            submission_rows.append([row_id] + vals)

    columns = ["id_seqpos"] + Config.TARGET_COLS
    submission_df = pd.DataFrame(submission_rows, columns=columns)

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")


def run_pipeline():
    """
    Runs the full training and submission pipeline.
    """
    train_model()
    generate_submission()
