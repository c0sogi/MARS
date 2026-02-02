import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import time
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import set_seed, MCRMSELoss, compute_global_mcrmse
from library.dataset import get_dataloader
from library.model import SPMHABiGRU


def train_epoch(model, loader, criterion, optimizer, device, grad_clip):
    """
    Performs one epoch of training.

    Args:
        model: The neural network model.
        loader: DataLoader for training data.
        criterion: Loss function (MCRMSELoss).
        optimizer: Optimizer (AdamW).
        device: Torch device.
        grad_clip: Maximum norm for gradient clipping.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        inputs = batch["inputs"].to(device)
        pair_indices = batch["pair_indices"].to(device)
        targets = batch["targets"].to(device)
        mask = batch["mask"].to(device)  # (B, L)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs, pair_indices)  # (B, L, 5)

        # Masking for Loss Calculation
        # The MCRMSELoss averages over dimensions (0, 1).
        # To strictly compute loss only on valid positions (first 68 bases),
        # we filter the flattened tensors using the mask.
        active_mask = mask > 0

        # Safety check for empty batches (unlikely given dataset structure)
        if active_mask.sum() == 0:
            continue

        # Select only valid positions and reshape to (1, N_valid, 5)
        # This trick allows MCRMSELoss to compute the mean over the valid set
        # while satisfying its expectation of input dimensions.
        masked_outputs = outputs[active_mask].unsqueeze(0)
        masked_targets = targets[active_mask].unsqueeze(0)

        loss = criterion(masked_outputs, masked_targets)

        if torch.isnan(loss):
            print("Warning: NaN loss detected. Skipping batch.")
            continue

        loss.backward()

        # Gradient Clipping (Critical for BiGRU stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def evaluate(model, loader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The neural network model.
        loader: DataLoader for validation data.
        device: Torch device.

    Returns:
        tuple: (global_mcrmse, scored_mcrmse)
    """
    model.eval()
    all_preds = []
    all_targets = []

    # Scored columns indices: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_indices = [0, 1, 3]

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(inputs, pair_indices)

            # Move to CPU
            preds_np = outputs.cpu().numpy()
            targets_np = targets.cpu().numpy()

            # Slice to scored length (68) as per competition metric definition
            preds_sliced = preds_np[:, : Config.PRED_LEN, :]
            targets_sliced = targets_np[:, : Config.PRED_LEN, :]

            all_preds.append(preds_sliced)
            all_targets.append(targets_sliced)

    # Compute metrics using global concatenation
    # Global MCRMSE (all 5 columns)
    global_mcrmse = compute_global_mcrmse(all_preds, all_targets)

    # Scored MCRMSE (3 specific columns used for leaderboard)
    scored_mcrmse = compute_global_mcrmse(
        all_preds, all_targets, target_indices=scored_indices
    )

    return global_mcrmse, scored_mcrmse


def train_model():
    """
    Main training routine. Initializes model, optimizer, scheduler, and runs the training loop.
    Implements Early Stopping and saves the best model.
    """
    # Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Device: {device}")

    # DataLoaders
    # We use cached data if available
    train_loader = get_dataloader("train", shuffle=True)
    val_loader = get_dataloader("val", shuffle=False)

    # Model
    model = SPMHABiGRU().to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR)

    # Loss
    criterion = MCRMSELoss()

    # Tracking
    best_score = float("inf")
    patience_counter = 0
    history = []

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, Config.GRAD_CLIP
        )

        # Validate
        val_loss, val_score = evaluate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        elapsed = time.time() - start_time

        # Print metrics (Full precision)
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.8f} | "
            f"Val MCRMSE (All): {val_loss:.8f} | "
            f"Val MCRMSE (Scored): {val_score:.8f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_score": val_score,
            }
        )

        # Checkpoint & Early Stopping
        # We optimize based on 'val_score' (the 3 scored columns)
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  >>> New Best Model Saved! Score: {best_score:.8f}")
        else:
            patience_counter += 1
            print(f"  >>> Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    return history


def predict_and_submit():
    """
    Loads the best model, generates predictions for the test set,
    and creates the submission CSV file.
    """
    device = torch.device(Config.DEVICE)

    # Load Test Data
    test_loader = get_dataloader("test", shuffle=False)

    # Load Model
    model = SPMHABiGRU().to(device)
    if not os.path.exists(Config.MODEL_PATH):
        print("No model checkpoint found. Skipping submission.")
        return

    print(f"Loading model from {Config.MODEL_PATH}...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    print("Generating predictions...")

    ids_list = []
    preds_list = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            ids = batch["id"]

            # Forward pass
            outputs = model(inputs, pair_indices)  # (B, 107, 5)

            preds_np = outputs.cpu().numpy()

            ids_list.extend(ids)
            preds_list.append(preds_np)

    # Concatenate all predictions
    # Shape: (N_samples, 107, 5)
    all_preds = np.concatenate(preds_list, axis=0)

    # Prepare Submission DataFrame
    # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

    submission_rows = []
    target_cols = Config.TARGET_COLS

    for i, sample_id in enumerate(ids_list):
        sample_preds = all_preds[i]  # (107, 5)

        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_preds = sample_preds[seqpos]

            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = float(row_preds[col_idx])

            submission_rows.append(row_dict)

    submission_df = pd.DataFrame(submission_rows)

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Submission shape: {submission_df.shape}")


def run():
    """
    Convenience function to run the full pipeline.
    """
    train_model()
    predict_and_submit()
