import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import seed_everything, mcrmse_loss, format_submission
from library.dataset import RNADataset
from library.model import StructureShortcutResBiGRU


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        # Move inputs to device
        sequence = batch["sequence"].to(device)
        loop_type = batch["loop_type"].to(device)
        pair_index = batch["pair_index"].to(device)
        pair_dist = batch["pair_dist"].to(device)
        targets = batch["targets"].to(device)  # Shape: (B, 68, 3)

        optimizer.zero_grad()

        # Forward pass
        # Output shape: (B, 107, 3)
        outputs = model(sequence, loop_type, pair_index, pair_dist)

        # Slice outputs to match the scored length (first 68 positions)
        # Targets are already shape (B, 68, 3)
        outputs_scored = outputs[:, : Config.PRED_LEN, :]

        # Compute Loss (MSE)
        loss = criterion(outputs_scored, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        running_loss += loss.item() * sequence.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set using MCRMSE.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            sequence = batch["sequence"].to(device)
            loop_type = batch["loop_type"].to(device)
            pair_index = batch["pair_index"].to(device)
            pair_dist = batch["pair_dist"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(sequence, loop_type, pair_index, pair_dist)

            # Slice to scored positions
            outputs_scored = outputs[:, : Config.PRED_LEN, :]

            all_preds.append(outputs_scored.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all batches
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    # Compute MCRMSE
    score = mcrmse_loss(y_true, y_pred)
    return score


def train_model():
    """
    Main function to train the model with Early Stopping.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Training on device: {device}")

    # 1. Load Data
    print("Loading datasets...")
    train_dataset = RNADataset(split="train", load_cached_data=True)
    val_dataset = RNADataset(split="val", load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Initialize Model
    model = StructureShortcutResBiGRU()
    model.to(device)

    # 3. Optimizer & Scheduler
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # 4. Loss Function (MSE)
    criterion = nn.MSELoss()

    # 5. Training Loop
    best_mcrmse = float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_mcrmse = validate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()

        elapsed = time.time() - start_time

        # Print metrics (Full precision)
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss (MSE): {train_loss:.6f} | "
            f"Val MCRMSE: {val_mcrmse} | "
            f"Time: {elapsed:.2f}s"
        )

        # Early Stopping & Checkpointing
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  -> New best model saved! MCRMSE: {best_mcrmse}")
        else:
            patience_counter += 1
            print(
                f"  -> No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
            )
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Validation MCRMSE: {best_mcrmse}")


def predict_and_submit():
    """
    Loads the best model, runs inference on the test set, and generates the submission file.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Load Data
    print("Loading test dataset...")
    test_dataset = RNADataset(split="test", load_cached_data=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Load Model
    print(f"Loading best model from {Config.MODEL_SAVE_PATH}...")
    model = StructureShortcutResBiGRU()
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()

    # 3. Inference
    all_preds = []
    all_ids = []

    print("Running inference on test set...")
    with torch.no_grad():
        for batch in test_loader:
            sequence = batch["sequence"].to(device)
            loop_type = batch["loop_type"].to(device)
            pair_index = batch["pair_index"].to(device)
            pair_dist = batch["pair_dist"].to(device)
            ids = batch["id"]

            # Forward pass
            outputs = model(sequence, loop_type, pair_index, pair_dist)

            # For submission, we need predictions for the full sequence length (107).
            # The model outputs (B, 107, 3), which is exactly what we need.
            # However, the submission format requires filling unscored positions too.
            # The prompt says: "Positions greater than the seq_scored value ... still need a value".
            # Our model predicts for all 107 positions. We will use these predictions.

            all_preds.append(outputs.cpu().numpy())
            all_ids.extend(ids)

    # Concatenate
    predictions = np.concatenate(all_preds, axis=0)  # Shape (N_test, 107, 3)

    # 4. Format Submission
    print("Generating submission file...")
    format_submission(all_ids, predictions, save_path=Config.SUBMISSION_FILE_PATH)
