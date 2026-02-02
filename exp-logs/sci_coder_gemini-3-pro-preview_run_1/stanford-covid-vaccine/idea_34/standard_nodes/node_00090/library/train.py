import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import seed_everything, mcrmse_loss, save_checkpoint, load_checkpoint
from library.data import get_dataloaders
from library.model import BondAwareModel


def train_one_epoch(model, loader, optimizer, device, criterion):
    """
    Trains the model for one epoch.
    Computes masked MSE loss on the first 68 positions.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Move data to device
        seq = batch["seq"].to(device)
        loop = batch["loop"].to(device)
        bond = batch["bond"].to(device)
        dist = batch["dist"].to(device)
        targets = batch["targets"].to(device)  # Shape: (B, 68, 3)

        optimizer.zero_grad()

        # Forward pass
        preds = model(seq, loop, bond, dist)  # Shape: (B, 107, 3)

        # Slice predictions to match target length (68)
        # Targets are already length 68 based on data.py processing,
        # but we ensure alignment with Config.pred_len
        preds_scored = preds[:, : Config.pred_len, :]
        targets_scored = targets[:, : Config.pred_len, :]

        # Compute Loss (MSE)
        loss = criterion(preds_scored, targets_scored)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Computes MCRMSE on the first 68 positions.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            bond = batch["bond"].to(device)
            dist = batch["dist"].to(device)
            targets = batch["targets"].to(device)

            preds = model(seq, loop, bond, dist)

            # Slice to scored length
            preds_scored = preds[:, : Config.pred_len, :]
            targets_scored = targets[:, : Config.pred_len, :]

            all_preds.append(preds_scored)
            all_targets.append(targets_scored)

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute MCRMSE
    score = mcrmse_loss(all_targets, all_preds)
    return score.item()


def run_training():
    """
    Main training loop.
    Handles initialization, training epochs, validation, and model saving.
    """
    seed_everything(Config.seed)

    # Setup directories
    os.makedirs(Config.working_dir, exist_ok=True)

    # Data Loaders
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(debug=Config.debug)

    # Model Setup
    print("Initializing Model...")
    device = torch.device(Config.device)
    model = BondAwareModel().to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.epochs, eta_min=Config.min_lr)

    # Loss Function (Standard MSE for training)
    criterion = nn.MSELoss()

    best_score = float("inf")
    best_epoch = 0

    # Early Stopping parameters
    patience = 5
    patience_counter = 0

    print(f"Starting training for {Config.epochs} epochs on {device}...")

    for epoch in range(1, Config.epochs + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)

        # Validate
        val_score = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch}/{Config.epochs} | "
            f"Train Loss (MSE): {train_loss:.6f} | "
            f"Val MCRMSE: {val_score:.10f} | "
            f"LR: {current_lr:.2e}"
        )

        # Save Best Model
        if val_score < best_score:
            best_score = val_score
            best_epoch = epoch
            save_checkpoint(model, optimizer, epoch, val_score, Config.model_path)
            print(f"  >>> New Best Model Saved! (Score: {best_score:.10f})")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs without improvement."
            )
            break

    print(f"Training complete. Best MCRMSE: {best_score:.10f} at Epoch {best_epoch}")


def generate_submission():
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("Generating Submission...")

    # Load Data
    _, _, test_loader = get_dataloaders(debug=Config.debug)

    # Load Model
    device = torch.device(Config.device)
    model = BondAwareModel().to(device)

    epoch, loss = load_checkpoint(model, None, Config.model_path, device=device)
    if epoch == 0:
        print("Warning: No checkpoint found. Using initialized model (random weights).")
    else:
        print(f"Loaded model from epoch {epoch} with val loss {loss:.6f}")

    model.eval()

    # Containers for submission data
    ids_list = []
    reactivity_list = []
    deg_Mg_pH10_list = []
    deg_pH10_list = []
    deg_Mg_50C_list = []
    deg_50C_list = []

    with torch.no_grad():
        for batch in test_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            bond = batch["bond"].to(device)
            dist = batch["dist"].to(device)
            batch_ids = batch["id"]

            # Forward pass (get full 107 length)
            preds = model(seq, loop, bond, dist)  # (B, 107, 3)
            preds = preds.cpu().numpy()

            # Iterate through batch
            for i, sample_id in enumerate(batch_ids):
                sample_preds = preds[i]  # (107, 3)

                # Targets trained: [reactivity, deg_Mg_pH10, deg_Mg_50C]
                # Indices: 0, 1, 2

                for seqpos in range(Config.seq_len):
                    # Construct ID
                    ids_list.append(f"{sample_id}_{seqpos}")

                    # Map predictions
                    reactivity_list.append(sample_preds[seqpos, 0])
                    deg_Mg_pH10_list.append(sample_preds[seqpos, 1])
                    deg_Mg_50C_list.append(sample_preds[seqpos, 2])

                    # Fill untrained columns with 0.0
                    deg_pH10_list.append(0.0)
                    deg_50C_list.append(0.0)

    # Create DataFrame
    submission_df = pd.DataFrame(
        {
            "id_seqpos": ids_list,
            "reactivity": reactivity_list,
            "deg_Mg_pH10": deg_Mg_pH10_list,
            "deg_pH10": deg_pH10_list,
            "deg_Mg_50C": deg_Mg_50C_list,
            "deg_50C": deg_50C_list,
        }
    )

    # Save
    submission_df.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")
    print(f"Submission shape: {submission_df.shape}")
