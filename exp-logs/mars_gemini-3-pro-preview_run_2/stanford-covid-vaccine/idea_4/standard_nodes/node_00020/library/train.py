import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library import config, utils, data, model


def train_fn(model_instance, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model_instance.train()
    running_loss = 0.0
    dataset_size = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        # Inputs: (B, 107, 18) -> Outputs: (B, 107, 5)
        outputs = model_instance(inputs)

        # Loss calculation
        # Criterion handles slicing outputs to match targets (68 positions)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping to prevent exploding gradients in RNN
        torch.nn.utils.clip_grad_norm_(model_instance.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        dataset_size += inputs.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def eval_fn(model_instance, loader, criterion, device):
    """
    Performs evaluation on the validation set.
    """
    model_instance.eval()

    # Accumulate squared errors per column to correctly compute global RMSE
    scored_indices = config.SCORED_INDICES
    num_scored = len(scored_indices)
    total_sse = torch.zeros(num_scored, device=device)
    total_elements = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model_instance(inputs)

            # Slice predictions to match targets (68 positions)
            if outputs.shape[1] > targets.shape[1]:
                outputs = outputs[:, : targets.shape[1], :]

            # Calculate squared errors
            diff = outputs - targets
            squared_errors = diff**2

            # Sum squared errors for scored columns over batch and sequence
            # Shape: (Batch, Seq, Channels) -> (Channels,)
            batch_sse = squared_errors[:, :, scored_indices].sum(dim=(0, 1))

            total_sse += batch_sse
            # Count elements per column: Batch * Seq_Len
            total_elements += inputs.size(0) * targets.size(1)

    # Calculate Global RMSE per column
    # MSE = Sum Squared Errors / Total Elements
    global_mse = total_sse / total_elements
    global_rmse = torch.sqrt(global_mse)

    # Calculate MCRMSE (Mean across columns)
    final_metric = torch.mean(global_rmse).item()

    return final_metric


def run_training():
    """
    Main function to orchestrate the training process.
    """
    # 1. Setup
    utils.seed_everything(config.SEED)
    device = torch.device(config.DEVICE)

    print(f"Device: {device}")
    print(f"Model Save Path: {config.MODEL_SAVE_PATH}")

    # 2. Data Loaders
    # Using load_cached_data=True to utilize the caching mechanism in data.py
    train_loader, val_loader, _ = data.get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    net = model.PartnerAwareHybridNet()
    net.to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(net.parameters(), lr=config.LEARNING_RATE)

    # Scheduler: Reduce LR if validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # 5. Loss Function
    criterion = utils.MCRMSELoss()

    # 6. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_fn(net, train_loader, optimizer, criterion, device)
        val_loss = eval_fn(net, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_loss)

        # Print metrics (Full precision for validation loss)
        print(
            f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss}"
        )

        # Early Stopping & Model Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(net.state_dict(), config.MODEL_SAVE_PATH)
            print(f"  New best model saved with Val Loss: {best_val_loss}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{config.PATIENCE}")

        if patience_counter >= config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Loss: {best_val_loss}")
