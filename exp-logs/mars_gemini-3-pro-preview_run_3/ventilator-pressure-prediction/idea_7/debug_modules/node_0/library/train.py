import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config, set_seed
from library.utils import get_device
from library.dataset import prepare_data
from library.model import FCPNet


def masked_mae_loss(y_pred, y_true, u_out):
    """
    Calculates Mean Absolute Error (L1 Loss) strictly for the inspiratory phase.

    Args:
        y_pred (torch.Tensor): Predicted pressure, shape (B, Seq_Len, 1) or (B, Seq_Len)
        y_true (torch.Tensor): Actual pressure, shape (B, Seq_Len)
        u_out (torch.Tensor): Expiratory valve status, shape (B, Seq_Len)

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Ensure shapes match
    y_pred = y_pred.squeeze(-1)

    # Create mask: 1 where u_out == 0 (Inspiratory), 0 otherwise
    mask = u_out == 0

    # Calculate absolute error
    error = torch.abs(y_pred - y_true)

    # Apply mask
    masked_error = error[mask]

    # Return mean of masked errors
    # Add a small epsilon to denominator to prevent nan if batch has no inspiratory phase (unlikely)
    if masked_error.numel() == 0:
        return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

    return masked_error.mean()


def train_epoch(model, loader, optimizer, device):
    """
    Runs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    counter = 0

    # u_out is at index 2 in Config.FEATURE_COLS
    U_OUT_IDX = 2

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Extract u_out for masking
        # inputs shape: (Batch, Seq, Features)
        u_out = inputs[:, :, U_OUT_IDX]

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute Loss
        loss = masked_mae_loss(outputs, targets, u_out)

        # Backward pass
        loss.backward()

        # Gradient clipping (optional but recommended for LSTMs)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()
        counter += 1

    return running_loss / counter


def validate(model, loader, device):
    """
    Runs validation loop.
    """
    model.eval()
    running_loss = 0.0
    counter = 0

    U_OUT_IDX = 2

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            u_out = inputs[:, :, U_OUT_IDX]

            outputs = model(inputs)
            loss = masked_mae_loss(outputs, targets, u_out)

            running_loss += loss.item()
            counter += 1

    return running_loss / counter


def generate_submission(model, loader, device):
    """
    Generates predictions for the test set and saves to submission.csv.
    """
    print("Generating predictions...")
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs in loader:
            inputs = inputs.to(device)
            # Forward pass
            preds = model(inputs)
            # Flatten: (Batch, Seq, 1) -> (Batch * Seq)
            preds = preds.squeeze(-1).flatten().cpu().numpy()
            predictions.extend(preds)

    predictions = np.array(predictions)

    # Load test metadata to get IDs
    # Note: prepare_data sorts by breath_id and time_step, so we must do the same
    # to ensure alignment.
    print("Loading test metadata for ID alignment...")
    test_df = pd.read_csv(Config.TEST_FILE)
    test_df = test_df.sort_values(["breath_id", "time_step"])

    if len(test_df) != len(predictions):
        raise ValueError(
            f"Shape mismatch: Metadata has {len(test_df)} rows, predictions have {len(predictions)}"
        )

    # Create submission DataFrame
    submission = pd.DataFrame({"id": test_df["id"], "pressure": predictions})

    # Save
    print(f"Saving submission to {Config.SUBMISSION_PATH}")
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")


def run_training(debug=False, load_cached_data=True):
    """
    Main orchestrator function.
    """
    set_seed(Config.SEED)
    device = get_device()
    print(f"Using device: {device}")

    # 1. Prepare Data
    train_loader, val_loader, test_loader = prepare_data(
        debug=debug, load_cached_data=load_cached_data
    )

    # 2. Initialize Model
    model = FCPNet(config=Config).to(device)

    # 3. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.FACTOR,
        patience=5,  # Scheduler patience < Early Stopping patience
        verbose=True,
        min_lr=Config.MIN_LR,
    )

    # 4. Training Loop
    best_val_loss = float("inf")
    early_stop_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss} | Val Loss: {val_loss}"
        )

        # Scheduler Step
        scheduler.step(val_loss)

        # Checkpointing & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"New best model saved with Val Loss: {val_loss}")
        else:
            early_stop_counter += 1
            print(
                f"No improvement. Early stopping counter: {early_stop_counter}/{Config.PATIENCE}"
            )

        if early_stop_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 5. Load Best Model
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    # 6. Generate Submission
    generate_submission(model, test_loader, device)


if __name__ == "__main__":
    # Default execution
    run_training()
