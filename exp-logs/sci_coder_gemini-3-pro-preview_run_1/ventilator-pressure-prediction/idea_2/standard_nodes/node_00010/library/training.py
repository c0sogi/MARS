import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import time

from library.config import Config
from library.utils import seed_everything
from library.data_processing import DataProcessor
from library.dataset import VentilatorDataset
from library.model import HybridCNNLSTM


class MaskedL1Loss(nn.Module):
    """
    Computes Mean Absolute Error (L1 Loss) strictly for the inspiratory phase.
    The expiratory phase is masked out using u_out (where u_out=1).
    """

    def __init__(self):
        super().__init__()

    def forward(self, preds, targets, u_out):
        """
        Args:
            preds (torch.Tensor): Predictions of shape (batch, seq_len).
            targets (torch.Tensor): Ground truth pressures of shape (batch, seq_len).
            u_out (torch.Tensor): Expiratory mask of shape (batch, seq_len).
                                  0 indicates inspiratory (valid), 1 indicates expiratory.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Create mask: 1 for inspiratory (keep), 0 for expiratory (discard)
        mask = 1.0 - u_out

        # Calculate absolute error
        absolute_error = torch.abs(preds - targets)

        # Apply mask
        masked_error = absolute_error * mask

        # Compute mean over valid elements
        # We add a small epsilon to the denominator to prevent division by zero,
        # though practically every breath has an inspiratory phase.
        loss = masked_error.sum() / (mask.sum() + 1e-8)

        return loss


def train_epoch(model, loader, optimizer, scheduler, criterion, device, epoch):
    """
    Runs one epoch of training.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    start_time = time.time()

    for batch_idx, (X, y, u_out) in enumerate(loader):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        u_out = u_out.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass
        preds = model(X)

        # Compute loss
        loss = criterion(preds, y, u_out)

        # Backward pass
        loss.backward()

        # Gradient clipping to prevent exploding gradients in LSTM
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimization step
        optimizer.step()

        # Scheduler step (OneCycleLR updates every batch)
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    duration = time.time() - start_time
    # Print basic info, full precision metrics reserved for validation
    print(f"Epoch {epoch} | Train Loss: {avg_loss} | Time: {duration:.2f}s")

    return avg_loss


def validate_epoch(model, loader, criterion, device, epoch):
    """
    Runs validation on the validation set.
    """
    model.eval()
    total_error_sum = 0.0
    total_valid_points = 0.0

    with torch.no_grad():
        for X, y, u_out in loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            u_out = u_out.to(device, non_blocking=True)

            preds = model(X)

            # Calculate exact MAE over the entire validation set
            mask = 1.0 - u_out
            abs_err = torch.abs(preds - y)
            masked_err = abs_err * mask

            total_error_sum += masked_err.sum().item()
            total_valid_points += mask.sum().item()

    # Compute global mean absolute error
    val_score = total_error_sum / (total_valid_points + 1e-8)

    # Print full precision as requested
    print(f"Epoch {epoch} | Val Loss: {val_score}")

    return val_score


def generate_submission(model, device, processor):
    """
    Generates predictions for the test set and saves to submission.csv.
    """
    print("Loading test data...")
    X_test, u_out_test = processor.load_dataset(split="test", load_cached_data=True)

    test_dataset = VentilatorDataset(X_test, u_out_test, y=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    model.eval()
    all_preds = []

    print("Generating predictions...")
    with torch.no_grad():
        for X, _ in test_loader:
            X = X.to(device)
            preds = model(X)
            all_preds.append(preds.cpu().numpy())

    # Concatenate all predictions: Shape (N_breaths, Seq_Len)
    predictions = np.concatenate(all_preds, axis=0)

    # Flatten to match the sample submission format (row-wise)
    # The data was processed sorted by breath_id and time_step, which matches sample submission
    predictions_flat = predictions.flatten()

    # Load sample submission to get IDs
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    sub_df = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

    # Ensure lengths match
    if len(sub_df) != len(predictions_flat):
        print(
            f"Warning: Submission length mismatch. Expected {len(sub_df)}, got {len(predictions_flat)}."
        )
        # Truncate or pad if necessary, though strictly should match
        min_len = min(len(sub_df), len(predictions_flat))
        sub_df = sub_df.iloc[:min_len]
        predictions_flat = predictions_flat[:min_len]

    sub_df[Config.COL_PRESSURE] = predictions_flat
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")


def run_training():
    """
    Main function to execute the training pipeline.
    """
    # 1. Reproducibility
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Preparation
    processor = DataProcessor()

    # Load Train
    X_train, y_train, u_out_train = processor.load_dataset(
        split="train", load_cached_data=True
    )
    train_dataset = VentilatorDataset(X_train, u_out_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    # Load Validation
    X_val, y_val, u_out_val = processor.load_dataset(split="val", load_cached_data=True)
    val_dataset = VentilatorDataset(X_val, u_out_val, y_val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = HybridCNNLSTM().to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    criterion = MaskedL1Loss()

    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.PCT_START,
        anneal_strategy="cos",
    )

    # 5. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_epoch(model, train_loader, optimizer, scheduler, criterion, device, epoch)

        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device, epoch)

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            print(
                f"Validation loss improved from {best_val_loss} to {val_loss}. Saving model..."
            )
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"Validation loss did not improve. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training finished. Best Validation Loss: {best_val_loss}")

    # 6. Submission Generation
    # Load best model
    print("Loading best model for submission...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    generate_submission(model, device, processor)
