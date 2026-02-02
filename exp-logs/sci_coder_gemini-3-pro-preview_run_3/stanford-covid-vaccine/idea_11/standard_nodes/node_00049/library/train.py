import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import seed_everything, MCRMSELoss, mcrmse
from library.data import get_dataloaders
from library.model import RNAModel


def train_one_epoch(model, loader, criterion, optimizer, device, config):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch in loader:
        inputs = batch["inputs"].to(device)
        pair_index = batch["pair_index"].to(device)
        targets = batch["targets"].to(device)
        mask = batch["mask"].to(device)

        batch_size = inputs.size(0)

        optimizer.zero_grad()

        # Forward pass: (B, 107, 5)
        preds = model(inputs, pair_index)

        # Apply mask to select only scored positions for loss calculation
        # mask is (B, 107), targets is (B, 107, 5)
        active_mask = mask.bool()

        # Flattened valid predictions and targets: (N_valid, 5)
        active_preds = preds[active_mask]
        active_targets = targets[active_mask]

        # Calculate Loss
        loss = criterion(active_preds, active_targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)

        optimizer.step()

        # Accumulate loss (weighted by batch size for reporting)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / total_samples


def validate(model, loader, device, config):
    """
    Evaluates the model on the validation set using global aggregation.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            pair_index = batch["pair_index"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)

            preds = model(inputs, pair_index)

            # Apply mask to extract valid scored positions
            active_mask = mask.bool()

            # Collect numpy arrays
            p = preds[active_mask].cpu().numpy()
            t = targets[active_mask].cpu().numpy()

            all_preds.append(p)
            all_targets.append(t)

    # Concatenate all batches to calculate global metric
    global_preds = np.concatenate(all_preds, axis=0)
    global_targets = np.concatenate(all_targets, axis=0)

    # Filter for scored columns
    global_preds = global_preds[..., config.SCORED_COLS_INDICES]
    global_targets = global_targets[..., config.SCORED_COLS_INDICES]

    # Calculate MCRMSE
    score = mcrmse(global_targets, global_preds)
    return score


def generate_submission(model, loader, device, config):
    """
    Generates predictions for the test set and formats the submission file.
    """
    model.eval()
    ids = []
    preds_list = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            pair_index = batch["pair_index"].to(device)
            batch_ids = batch["id"]

            # Forward pass
            preds = model(inputs, pair_index)  # (B, 107, 5)
            preds = preds.cpu().numpy()

            for i in range(len(batch_ids)):
                ids.append(batch_ids[i])
                preds_list.append(preds[i])

    # Format data for CSV
    submission_data = []
    cols = config.target_cols

    for sample_idx, sample_id in enumerate(ids):
        sample_preds = preds_list[sample_idx]  # (107, 5)

        # Create a row for each sequence position
        for seqpos in range(config.seq_len):
            row_id = f"{sample_id}_{seqpos}"
            row_values = sample_preds[seqpos]

            row_dict = {"id_seqpos": row_id}
            for col_name, val in zip(cols, row_values):
                row_dict[col_name] = val
            submission_data.append(row_dict)

    return pd.DataFrame(submission_data)


def run_training(epochs=None, batch_size=None, debug=False):
    """
    Main execution function for training and submission generation.
    """
    # Initialize Config
    config = Config()

    # Apply overrides
    if epochs is not None:
        config.epochs = epochs
    if batch_size is not None:
        config.batch_size = batch_size
    if debug:
        config.debug = True
        config.epochs = 2  # Short run for debugging

    seed_everything(config.seed)

    print(f"Initializing training with device: {config.device}")

    # Load Data (with caching)
    train_loader, val_loader, test_loader = get_dataloaders(
        config, load_cached_data=True
    )

    # Initialize Model
    model = RNAModel(config).to(config.device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.eta_min
    )

    # Loss Function
    criterion = MCRMSELoss()

    # Training Loop Variables
    best_score = float("inf")
    patience_counter = 0

    print("Starting training loop...")

    for epoch in range(config.epochs):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, config.device, config
        )

        # Validate
        val_score = validate(model, val_loader, config.device, config)

        # Step Scheduler
        scheduler.step()

        # Print Metrics
        print(
            f"Epoch {epoch+1}/{config.epochs} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score}"
        )

        # Checkpointing
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), config.model_save_path)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= config.patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training complete. Best Validation MCRMSE: {best_score}")

    # Inference
    print("Loading best model for inference...")
    model.load_state_dict(
        torch.load(config.model_save_path, map_location=config.device)
    )

    print("Generating submission...")
    df_sub = generate_submission(model, test_loader, config.device, config)

    df_sub.to_csv(config.submission_path, index=False)
    print(f"Submission saved to {config.submission_path}")
