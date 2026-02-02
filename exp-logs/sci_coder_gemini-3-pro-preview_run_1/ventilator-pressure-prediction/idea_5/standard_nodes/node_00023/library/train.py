import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from library import config, utils, data, model


class MaskedL1Loss(nn.Module):
    """
    Computes the Mean Absolute Error (L1 Loss) strictly for the inspiratory phase.
    The expiratory phase (where u_out == 1) is masked out.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target, u_out):
        """
        Args:
            pred (Tensor): Predicted pressure (Batch, Seq)
            target (Tensor): Actual pressure (Batch, Seq)
            u_out (Tensor): Expiratory valve control (Batch, Seq), 0=In, 1=Out

        Returns:
            Tensor: Scalar loss value (Mean Absolute Error over valid steps)
        """
        # Create mask: 1 for inspiratory (u_out=0), 0 for expiratory (u_out=1)
        mask = 1 - u_out

        # Calculate absolute error
        error = torch.abs(pred - target)

        # Apply mask
        masked_error = error * mask

        # Calculate mean over valid elements
        # Add a small epsilon to denominator to prevent division by zero
        loss = masked_error.sum() / (mask.sum() + 1e-8)

        return loss


def train_epoch(model, loader, optimizer, scheduler, device, criterion):
    """
    Runs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    total_batches = 0

    for batch in loader:
        # Move data to device
        x_cont = batch["x_cont"].to(device)
        u_out = batch["u_out"].to(device)
        y = batch["y"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        preds = model(x_cont)

        # Compute loss
        loss = criterion(preds, y, u_out)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)

        # Optimization step
        optimizer.step()
        scheduler.step()

        # Accumulate loss
        running_loss += loss.item()
        total_batches += 1

    return running_loss / total_batches


def validate_epoch(model, loader, device, criterion):
    """
    Runs validation and calculates the global MAE over the inspiratory phase.
    """
    model.eval()

    total_error_sum = 0.0
    total_mask_sum = 0.0

    with torch.no_grad():
        for batch in loader:
            x_cont = batch["x_cont"].to(device)
            u_out = batch["u_out"].to(device)
            y = batch["y"].to(device)

            preds = model(x_cont)

            # Calculate metrics for global MAE
            mask = 1 - u_out
            error = torch.abs(preds - y)

            total_error_sum += (error * mask).sum().item()
            total_mask_sum += mask.sum().item()

    # Calculate global MAE
    global_mae = total_error_sum / (total_mask_sum + 1e-8)

    return global_mae


def train_model():
    """
    Main function to train the model.
    """
    # 1. Setup
    utils.seed_everything()
    device = utils.get_device()
    print(f"Using device: {device}")

    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # 2. Data
    print("Loading data...")
    # load_cached_data=True ensures we use the cache logic in library.features
    train_loader, val_loader, _ = data.get_data_loaders(load_cached_data=True)

    # 3. Model
    print("Initializing model...")
    net = model.PhysicsResidualModel().to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # OneCycleLR configuration
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.MAX_LR,
        epochs=config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=config.PCT_START,
        anneal_strategy=config.ANNEAL_STRATEGY,
        div_factor=config.DIV_FACTOR,
        final_div_factor=config.FINAL_DIV_FACTOR,
    )

    criterion = MaskedL1Loss()

    # 5. Training Loop
    best_val_loss = float("inf")

    print(f"Starting training for {config.EPOCHS} epochs...")
    start_time = time.time()

    for epoch in range(config.EPOCHS):
        epoch_start = time.time()

        train_loss = train_epoch(
            net, train_loader, optimizer, scheduler, device, criterion
        )
        val_loss = validate_epoch(net, val_loader, device, criterion)

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MAE: {val_loss:.10f} | "
            f"Time: {epoch_time:.2f}s"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), config.MODEL_SAVE_PATH)
            print(f"New best model saved with Val MAE: {best_val_loss:.10f}")

    total_time = time.time() - start_time
    print(f"Training complete. Total time: {total_time:.2f}s")
    print(f"Best Validation MAE: {best_val_loss:.10f}")


def predict_and_submit():
    """
    Generates predictions on the test set using the best saved model and creates a submission file.
    """
    utils.seed_everything()
    device = utils.get_device()

    # Load data (only test loader needed)
    _, _, test_loader = data.get_data_loaders(load_cached_data=True)

    # Load Model
    net = model.PhysicsResidualModel().to(device)
    if not os.path.exists(config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model file not found at {config.MODEL_SAVE_PATH}. Train the model first."
        )

    print(f"Loading best model from {config.MODEL_SAVE_PATH}...")
    net.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    net.eval()

    print("Generating predictions...")

    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in test_loader:
            x_cont = batch["x_cont"].to(device)
            ids = batch["ids"]  # Keep on CPU

            preds = net(x_cont)

            # Flatten predictions and ids: (Batch, Seq) -> (Batch*Seq)
            preds_flat = preds.cpu().view(-1).numpy()
            ids_flat = ids.view(-1).numpy()

            all_preds.append(preds_flat)
            all_ids.append(ids_flat)

    # Concatenate all batches
    final_preds = np.concatenate(all_preds)
    final_ids = np.concatenate(all_ids)

    # Create DataFrame
    submission = pd.DataFrame(
        {config.ID_COL: final_ids, config.TARGET_COL: final_preds}
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)

    # Save
    submission.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
    print(f"Submission shape: {submission.shape}")
