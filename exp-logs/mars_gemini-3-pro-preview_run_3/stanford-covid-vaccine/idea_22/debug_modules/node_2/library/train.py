import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import set_seed, compute_mcrmse
from library.loss import MCRMSELoss
from library.data import get_dataloaders
from library.model import NonLinearChannelGatedBiGRU


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one training epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        inputs = batch["inputs"].to(device)
        pair_indices = batch["pair_indices"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs, pair_indices)

        # Slice outputs to match target length (68) for loss calculation
        # Note: Slicing creates a non-contiguous tensor, but MCRMSELoss uses .reshape()
        # which handles this safely (Cite debug_lesson_2).
        outputs_scored = outputs[:, : Config.PRED_LEN, :]

        # Compute loss on all 5 columns
        loss = criterion(outputs_scored, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping (Mandatory for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimizer step
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Aggregates predictions globally before computing MCRMSE.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(inputs, pair_indices)

            # Slice outputs to match target length (68)
            outputs_scored = outputs[:, : Config.PRED_LEN, :]

            loss = criterion(outputs_scored, targets)
            running_loss += loss.item() * inputs.size(0)

            # Store predictions and targets for global metric calculation
            all_preds.append(outputs_scored.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    val_loss = running_loss / len(loader.dataset)

    # Concatenate all batches to avoid averaging bias
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute MCRMSE on scored columns (reactivity, deg_Mg_pH10, deg_Mg_50C)
    # Indices: 0, 1, 3
    mcrmse = compute_mcrmse(all_preds, all_targets, scored_indices=[0, 1, 3])

    return val_loss, mcrmse


def generate_submission(model, loader, device, output_path):
    """
    Generates predictions for the test set and saves to CSV.
    """
    model.eval()
    all_preds = []
    all_ids = []

    print("Generating predictions for test set...")
    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            ids = batch["id"]

            outputs = model(inputs, pair_indices)

            all_preds.append(outputs.cpu().numpy())
            all_ids.extend(ids)

    # Shape: (N_samples, 107, 5)
    all_preds = np.concatenate(all_preds, axis=0)

    # Prepare submission data
    submission_data = []
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Flatten predictions to (N_samples * Seq_Len, Columns) format
    for i, sample_id in enumerate(all_ids):
        sample_preds = all_preds[i]  # (107, 5)
        for seq_pos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seq_pos}"
            row_preds = sample_preds[seq_pos]

            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = row_preds[col_idx]

            submission_data.append(row_dict)

    df_sub = pd.DataFrame(submission_data)

    # Ensure column order matches sample submission
    cols = ["id_seqpos"] + target_cols
    df_sub = df_sub[cols]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(debug=False, epochs=None, batch_size=None):
    """
    Main execution function for training the model.
    """
    # Initialize Config with overrides
    config_overrides = {}
    if epochs is not None:
        config_overrides["epochs"] = epochs
    if batch_size is not None:
        config_overrides["batch_size"] = batch_size

    cfg = Config(debug=debug, **config_overrides)

    # Set reproducibility
    set_seed(cfg.SEED)
    device = torch.device(cfg.DEVICE)

    print(f"Starting training on {device} (Debug={debug})")

    # Get DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=debug, batch_size=cfg.BATCH_SIZE
    )

    # Initialize Model
    model = NonLinearChannelGatedBiGRU().to(device)

    # Loss, Optimizer, Scheduler
    criterion = MCRMSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)

    # Tracking
    best_mcrmse = float("inf")
    best_epoch = -1
    patience = 5
    patience_counter = 0

    # Training Loop
    for epoch in range(cfg.EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mcrmse = validate(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1}/{cfg.EPOCHS} | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val MCRMSE: {val_mcrmse}"
        )

        # Save Best Model
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), cfg.MODEL_SAVE_PATH)
            print(f"New best model saved! (MCRMSE: {best_mcrmse})")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs without improvement."
            )
            break

    print(f"Training finished. Best MCRMSE: {best_mcrmse} at Epoch {best_epoch}")

    # Generate Submission using best model
    if os.path.exists(cfg.MODEL_SAVE_PATH):
        print("Loading best model for submission...")
        model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH, map_location=device))
        generate_submission(model, test_loader, device, cfg.SUBMISSION_PATH)
    else:
        print("Error: Model file not found. Submission not generated.")
