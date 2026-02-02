import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library import config
from library.loss import MaskedMCRMSELoss


def train_step(model, batch, optimizer, device, loss_fn):
    """
    Performs a single training step using the 2-pass recycling strategy.

    Args:
        model (nn.Module): The HI-GFDN model.
        batch (tuple): A tuple containing (x, partner_indices, targets).
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): The device to run on.
        loss_fn (nn.Module): The loss function (MaskedMCRMSELoss).

    Returns:
        float: The loss value for this step.
    """
    x, p, y = batch
    x = x.to(device)
    p = p.to(device)
    y = y.to(device)

    optimizer.zero_grad()

    # 1. Static Encoding
    # Compute Z once from static inputs
    z = model.static_encoder(x)

    # 2. Recycling Pass 1
    # Initialize previous predictions with zeros
    batch_size, _, length = x.shape
    y_prev_0 = torch.zeros((batch_size, 5, length), device=device, dtype=x.dtype)

    # Forward pass 1
    y_pred_1 = model.recurrent_decoder(z, y_prev_0, p)

    # 3. Recycling Pass 2
    # Detach gradients from Pass 1 output to stop gradient flow through the feedback mechanism
    y_prev_1 = y_pred_1.detach()

    # Forward pass 2
    y_pred_2 = model.recurrent_decoder(z, y_prev_1, p)

    # 4. Loss Calculation
    # Total Loss = MCRMSE(Y2) + 0.5 * MCRMSE(Y1)
    loss1 = loss_fn(y_pred_1, y)
    loss2 = loss_fn(y_pred_2, y)
    loss = loss2 + 0.5 * loss1

    # 5. Backpropagation
    loss.backward()
    optimizer.step()

    return loss.item()


def train_epoch(model, dataloader, optimizer, device, loss_fn):
    """
    Runs one epoch of training.

    Args:
        model (nn.Module): The model.
        dataloader (DataLoader): Training dataloader.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device.
        loss_fn (nn.Module): Loss function.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = len(dataloader)

    for batch in dataloader:
        loss = train_step(model, batch, optimizer, device, loss_fn)
        running_loss += loss

    return running_loss / num_batches if num_batches > 0 else 0.0


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set using Correct Global RMSE.
    Accumulates SSE and counts across the entire dataset before computing RMSE.

    Args:
        model (nn.Module): The model.
        dataloader (DataLoader): Validation dataloader.
        device (torch.device): Device.

    Returns:
        float: The MCRMSE score.
    """
    model.eval()

    # Scored columns: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_cols_indices = [0, 1, 3]
    scored_len = config.SCORED_LEN

    # Accumulators for SSE (Sum of Squared Errors) per column
    # Shape: (3,) corresponding to the 3 scored columns
    total_sse = torch.zeros(3, device=device)
    total_count = 0

    with torch.no_grad():
        for x, p, y in dataloader:
            x = x.to(device)
            p = p.to(device)
            y = y.to(device)

            # Inference: 2 Passes
            z = model.static_encoder(x)

            # Pass 1
            batch_size, _, length = x.shape
            y_prev = torch.zeros((batch_size, 5, length), device=device, dtype=x.dtype)
            y_pred_1 = model.recurrent_decoder(z, y_prev, p)

            # Pass 2 (Final Prediction)
            y_pred_2 = model.recurrent_decoder(z, y_pred_1, p)

            # Slice to valid scored region
            # Preds: (B, 3, 68)
            preds_masked = y_pred_2[:, scored_cols_indices, :scored_len]
            targets_masked = y[:, scored_cols_indices, :scored_len]

            # Calculate Squared Error
            # Sum over Batch (dim 0) and Sequence Length (dim 2)
            # Result shape: (3,)
            sse = torch.sum((preds_masked - targets_masked) ** 2, dim=(0, 2))

            total_sse += sse
            total_count += batch_size * scored_len

    # Compute RMSE per column: sqrt(Total SSE / Total Count)
    # Note: total_count is the total number of elements per column (N_samples * 68)
    rmse_per_col = torch.sqrt(total_sse / total_count)

    # MCRMSE is the mean of the column RMSEs
    mcrmse = torch.mean(rmse_per_col).item()

    return mcrmse


def run_training(
    model,
    train_loader,
    val_loader,
    device,
    epochs=config.EPOCHS,
    patience=config.PATIENCE,
):
    """
    Orchestrates the full training loop with early stopping and learning rate scheduling.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data.
        val_loader (DataLoader): Validation data.
        device (torch.device): Device.
        epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.
    """
    print(f"Starting training on {device}...")

    # Optimizer and Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )
    loss_fn = MaskedMCRMSELoss()

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    best_mcrmse = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        avg_train_loss = train_epoch(model, train_loader, optimizer, device, loss_fn)

        # Validate
        val_mcrmse = validate(model, val_loader, device)

        # Logging
        # Printing full precision as requested
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss} | Val MCRMSE: {val_mcrmse}"
        )

        # Scheduler Step
        scheduler.step(val_mcrmse)

        # Checkpointing & Early Stopping
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), config.MODEL_PATH)
            print(f"  New best model saved! ({val_mcrmse})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Val MCRMSE: {best_mcrmse}")
