import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.model import RPCNet


class MaskedL1Loss(nn.Module):
    """
    Mean Absolute Error (L1 Loss) calculated strictly on the inspiratory phase.
    The inspiratory phase is defined where u_out == 0.
    """

    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target, u_out):
        """
        Args:
            pred: Predicted pressure (Batch, Seq_Len, 1)
            target: Actual pressure (Batch, Seq_Len, 1)
            u_out: Control input u_out (Batch, Seq_Len, 1), 0=Inspiratory, 1=Expiratory
        """
        # Create mask: 1 where u_out is 0 (inspiratory), 0 otherwise
        mask = 1.0 - u_out

        # Calculate element-wise absolute error
        abs_err = torch.abs(pred - target)

        # Apply mask
        masked_err = abs_err * mask

        # Compute mean over valid elements
        # Add epsilon to denominator to prevent division by zero in edge cases
        loss = masked_err.sum() / (mask.sum() + 1e-8)

        return loss


def train_epoch(model, loader, optimizer, criterion, device, max_grad_norm):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in loader:
        # Move data to device
        X = batch["X"].to(device)
        y = batch["y"].to(device)
        u_out = batch["u_out"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        preds = model(X)

        # Compute masked loss
        loss = criterion(preds, y, u_out)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Optimizer step
        optimizer.step()

        # Accumulate metrics
        running_loss += loss.item()
        num_batches += 1

    return running_loss / num_batches


def validate_epoch(model, loader, criterion, device):
    """
    Performs one epoch of validation (no gradient updates).
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            X = batch["X"].to(device)
            y = batch["y"].to(device)
            u_out = batch["u_out"].to(device)

            preds = model(X)
            loss = criterion(preds, y, u_out)

            running_loss += loss.item()
            num_batches += 1

    return running_loss / num_batches


def predict(model, loader, device):
    """
    Generates predictions for the test set.
    Returns a flattened numpy array of predictions corresponding to the test CSV rows.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            X = batch["X"].to(device)
            # Note: Test loader might include u_out, but we don't need it for prediction
            # and we definitely don't have y.

            preds = model(X)

            # Move to CPU and convert to numpy
            # preds shape: (Batch, Seq_Len, 1)
            all_preds.append(preds.cpu().numpy())

    # Concatenate all batches: (N_samples, Seq_Len, 1)
    if len(all_preds) > 0:
        full_preds = np.concatenate(all_preds, axis=0)
        # Flatten to match the CSV format (Row-wise)
        return full_preds.flatten()
    else:
        return np.array([])


def run_training(train_loader, val_loader, test_loader):
    """
    Main execution function for training, validation, and submission generation.
    """
    # Ensure reproducibility
    torch.manual_seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Initializing model on {device}...")

    # Initialize Model
    model = RPCNet().to(device)

    # Initialize Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Initialize Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # Initialize Loss
    criterion = MaskedL1Loss()

    # Training Loop Variables
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, Config.MAX_GRAD_NORM
        )

        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)

        # Update Scheduler
        scheduler.step(val_loss)

        elapsed = time.time() - start_time

        # Print metrics (Full precision as requested)
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Time: {elapsed:.2f}s | "
            f"Train Loss: {train_loss} | Val Loss: {val_loss}"
        )

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT)
            print(f"Saved new best model to {Config.MODEL_CHECKPOINT}")
        else:
            patience_counter += 1
            print(f"EarlyStopping counter: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print("Training finished.")

    # --- Inference Phase ---
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT, map_location=device))

    print("Generating predictions...")
    flat_predictions = predict(model, test_loader, device)

    # --- Submission Generation ---
    print("Preparing submission file...")

    # Load test metadata to align IDs
    # We must ensure the order matches how data_utils processed the data
    test_df = pd.read_csv(Config.TEST_CSV)

    # data_utils.compute_features sorts by breath_id and time_step
    test_df = test_df.sort_values(["breath_id", "time_step"])

    # Sanity check
    if len(flat_predictions) != len(test_df):
        raise ValueError(
            f"Prediction count mismatch! Expected {len(test_df)}, got {len(flat_predictions)}"
        )

    # Assign predictions
    test_df["pressure"] = flat_predictions

    # Select columns required for submission
    submission_df = test_df[["id", "pressure"]]

    # Save to disk
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)

    print(f"Submission successfully saved to {Config.SUBMISSION_FILE}")
