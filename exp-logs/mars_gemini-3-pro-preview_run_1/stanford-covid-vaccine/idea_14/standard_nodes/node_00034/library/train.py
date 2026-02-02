import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import set_seed, mcrmse, save_submission
from library.dataset import get_dataloaders
from library.model import RNAModel
from library.loss import MaskedMSELoss


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    start_time = time.time()

    for batch_idx, batch in enumerate(loader):
        # Move data to device
        sequence = batch["sequence"].to(device)
        loop_type = batch["loop_type"].to(device)
        pair_dist = batch["pair_dist"].to(device)
        targets = batch["targets"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        preds = model(sequence, loop_type, pair_dist)

        # Compute loss
        loss = criterion(preds, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimizer step
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    elapsed = time.time() - start_time

    # Optional: Print epoch summary here if needed, but main loop handles logging usually.
    return avg_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and MCRMSE score.
    """
    model.eval()
    running_loss = 0.0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            sequence = batch["sequence"].to(device)
            loop_type = batch["loop_type"].to(device)
            pair_dist = batch["pair_dist"].to(device)
            targets = batch["targets"].to(device)

            # Forward pass
            preds = model(sequence, loop_type, pair_dist)

            # Compute loss
            loss = criterion(preds, targets)
            running_loss += loss.item()

            # Collect predictions and targets for MCRMSE calculation
            # We only score the first 68 positions (Config.PRED_LEN)
            # The MaskedMSELoss handles training, but for metric we slice manually
            preds_sliced = preds[:, : Config.PRED_LEN, :]
            targets_sliced = targets[:, : Config.PRED_LEN, :]

            all_preds.append(preds_sliced.cpu().numpy())
            all_targets.append(targets_sliced.cpu().numpy())

    avg_loss = running_loss / len(loader)

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate MCRMSE
    score = mcrmse(all_targets, all_preds)

    return avg_loss, score


def predict(model, loader, device):
    """
    Runs inference on the test set.
    Returns list of IDs and array of predictions.
    """
    model.eval()
    all_ids = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            sequence = batch["sequence"].to(device)
            loop_type = batch["loop_type"].to(device)
            pair_dist = batch["pair_dist"].to(device)
            ids = batch["id"]

            # Forward pass
            preds = model(sequence, loop_type, pair_dist)

            all_preds.append(preds.cpu().numpy())
            all_ids.extend(ids)

    # Concatenate predictions: (N_samples, 107, 3)
    all_preds = np.concatenate(all_preds, axis=0)

    return all_ids, all_preds


def run_training():
    """
    Main function to execute the training pipeline, evaluation, and submission generation.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    print("Initializing Model...")
    model = RNAModel().to(device)

    # 4. Optimizer, Scheduler, Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN)

    criterion = MaskedMSELoss()

    # 5. Training Loop
    best_mcrmse = float("inf")
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(1, Config.EPOCHS + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_mcrmse = validate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_duration = time.time() - epoch_start

        print(
            f"Epoch {epoch}/{Config.EPOCHS} | "
            f"Time: {epoch_duration:.1f}s | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val MCRMSE: {val_mcrmse:.10f}"
        )  # Full precision as requested

        # Early Stopping & Checkpointing
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  [+] New Best Model Saved! MCRMSE: {best_mcrmse:.10f}")
        else:
            patience_counter += 1
            print(
                f"  [-] No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 6. Inference
    print("\nLoading best model for inference...")
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Warning: No best model found. Using current model state.")

    print("Generating predictions on test set...")
    test_ids, test_preds = predict(model, test_loader, device)

    # 7. Submission
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    save_submission(test_ids, test_preds, Config.SUBMISSION_PATH)
    print("Done.")
