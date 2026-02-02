import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.model import HybridCNNLSTM


def masked_mae_loss(y_pred, y_true, u_out):
    """
    Computes Mean Absolute Error (MAE) only for the inspiratory phase.

    Args:
        y_pred (torch.Tensor): Predicted pressure of shape (Batch, Seq_Len).
        y_true (torch.Tensor): Actual pressure of shape (Batch, Seq_Len).
        u_out (torch.Tensor): Expiratory valve status of shape (Batch, Seq_Len).
                              0 = Inspiratory, 1 = Expiratory.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Create mask: 1 for inspiratory (u_out == 0), 0 for expiratory (u_out == 1)
    mask = 1 - u_out

    # Calculate absolute error
    absolute_error = torch.abs(y_pred - y_true)

    # Apply mask to error
    masked_error = absolute_error * mask

    # Compute sum of errors and sum of valid mask elements
    total_error = masked_error.sum()
    total_elements = mask.sum()

    # Avoid division by zero (though highly unlikely in a full batch)
    if total_elements < 1e-7:
        return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

    return total_error / total_elements


def train_one_epoch(model, dataloader, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network.
        dataloader (DataLoader): Training data loader.
        optimizer (Optimizer): PyTorch optimizer.
        device (torch.device): Compute device.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    # Get index of u_out feature to create mask
    # u_out is needed to compute the metric correctly
    feature_indices = Config.get_feature_indices()
    u_out_idx = feature_indices["u_out"]

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        # Extract u_out from input features: (Batch, Seq_Len)
        u_out = x[:, :, u_out_idx]

        optimizer.zero_grad()

        # Forward pass
        preds = model(x)

        # Compute masked loss
        loss = masked_mae_loss(preds, y, u_out)

        # Backward pass
        loss.backward()

        # Gradient clipping to prevent exploding gradients in LSTM
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, dataloader, device):
    """
    Evaluates the model on validation data.

    Args:
        model (nn.Module): The neural network.
        dataloader (DataLoader): Validation data loader.
        device (torch.device): Compute device.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    feature_indices = Config.get_feature_indices()
    u_out_idx = feature_indices["u_out"]

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            u_out = x[:, :, u_out_idx]

            preds = model(x)
            loss = masked_mae_loss(preds, y, u_out)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def predict(model, dataloader, device):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        model (nn.Module): The trained neural network.
        dataloader (DataLoader): Test data loader.
        device (torch.device): Compute device.
    """
    model.eval()
    all_preds = []

    print("Generating predictions on test set...")
    with torch.no_grad():
        for x in dataloader:
            x = x.to(device)
            preds = model(x)

            # preds shape: (Batch, Seq_Len)
            # Flatten to (Batch * Seq_Len) to match submission format (one row per time step)
            preds_flat = preds.cpu().numpy().flatten()
            all_preds.append(preds_flat)

    # Concatenate all batches
    final_preds = np.concatenate(all_preds)

    # Load test metadata to get the correct IDs
    print(f"Loading test metadata from {Config.TEST_DATA_PATH}...")
    test_df = pd.read_csv(Config.TEST_DATA_PATH)

    # Verify alignment
    if len(final_preds) != len(test_df):
        raise ValueError(
            f"Prediction count {len(final_preds)} does not match test set size {len(test_df)}"
        )

    # Create submission DataFrame
    submission = pd.DataFrame(
        {Config.ID_COL: test_df[Config.ID_COL], Config.TARGET_COL: final_preds}
    )

    # Save to disk
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)
    submission.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")


def run_training(train_loader, val_loader, test_loader, epochs=None, patience=None):
    """
    Orchestrates the training process, including initialization, training loop,
    early stopping, and final prediction.

    Args:
        train_loader (DataLoader): Training data.
        val_loader (DataLoader): Validation data.
        test_loader (DataLoader): Test data.
        epochs (int, optional): Override default number of epochs.
        patience (int, optional): Override default early stopping patience.
    """
    # 1. Setup Environment
    Config.initialize()
    device = torch.device(Config.DEVICE)
    print(f"Engine initialized. Device: {device}")

    # 2. Initialize Model
    model = HybridCNNLSTM().to(device)

    # 3. Setup Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.MIN_LR,
    )

    # 4. Training Loop
    num_epochs = epochs if epochs is not None else Config.EPOCHS
    early_stopping_patience = (
        patience if patience is not None else Config.EARLY_STOPPING_PATIENCE
    )

    best_val_loss = float("inf")
    patience_counter = 0

    print(
        f"Starting training for {num_epochs} epochs with patience {early_stopping_patience}..."
    )

    for epoch in range(num_epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validate
        val_loss = evaluate(model, val_loader, device)

        # Update Scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Log metrics (Full precision)
        print(
            f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | LR: {current_lr}"
        )

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), Config.MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(
                    f"Early stopping triggered at epoch {epoch+1}. Best Val Loss: {best_val_loss}"
                )
                break

    print(f"Training finished. Best Validation Loss: {best_val_loss}")

    # 5. Final Prediction
    print(f"Loading best model from {Config.MODEL_PATH} for inference...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    predict(model, test_loader, device)
