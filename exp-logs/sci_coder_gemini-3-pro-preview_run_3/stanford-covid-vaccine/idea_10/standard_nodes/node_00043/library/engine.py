import torch
import torch.nn as nn
import numpy as np
import os
import library.config
from library.utils import mcrmse_loss, seed_everything
from library.model import MaskedBiGRU


def train_fn(model, dataloader, optimizer, device, scheduler=None):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    running_reg_loss = 0.0
    running_recon_loss = 0.0
    dataset_size = 0

    # Reconstruction loss criterion (MSE for one-hot reconstruction)
    recon_criterion = nn.MSELoss()

    for batch_idx, (masked_input, original_input, targets, mask) in enumerate(
        dataloader
    ):
        masked_input = masked_input.to(device)
        original_input = original_input.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        batch_size = masked_input.size(0)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        # reg_out: (Batch, 107, 5)
        # recon_out: (Batch, 107, 14)
        reg_out, recon_out = model(masked_input)

        # --- 1. Main Task: Regression Loss ---
        # Only score the first SEQ_SCORED (68) positions
        # Targets are already padded, but we only care about valid positions for the metric
        reg_out_scored = reg_out[:, : library.config.Config.SEQ_SCORED, :]
        targets_scored = targets[:, : library.config.Config.SEQ_SCORED, :]

        loss_reg = mcrmse_loss(reg_out_scored, targets_scored)

        # --- 2. Auxiliary Task: Reconstruction Loss ---
        # Calculate loss only on masked positions
        # mask shape: (Batch, 107). 1.0 means masked.
        # Flatten for easier indexing
        mask_flat = mask.view(-1).bool()

        if mask_flat.sum() > 0:
            recon_out_flat = recon_out.view(-1, library.config.Config.INPUT_DIM)
            original_input_flat = original_input.view(
                -1, library.config.Config.INPUT_DIM
            )

            # Select only masked vectors
            recon_masked = recon_out_flat[mask_flat]
            original_masked = original_input_flat[mask_flat]

            loss_recon = recon_criterion(recon_masked, original_masked)
        else:
            loss_recon = torch.tensor(0.0, device=device)

        # --- 3. Composite Loss ---
        loss = loss_reg + (library.config.Config.RECON_LOSS_WEIGHT * loss_recon)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        running_loss += loss.item() * batch_size
        running_reg_loss += loss_reg.item() * batch_size
        running_recon_loss += loss_recon.item() * batch_size
        dataset_size += batch_size

    # Calculate average losses
    epoch_loss = running_loss / dataset_size
    epoch_reg_loss = running_reg_loss / dataset_size
    epoch_recon_loss = running_recon_loss / dataset_size

    return epoch_loss, epoch_reg_loss, epoch_recon_loss


def eval_fn(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    Returns the MCRMSE score.
    """
    model.eval()
    running_mcrmse = 0.0
    dataset_size = 0

    with torch.no_grad():
        for inputs, _, targets, _ in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            batch_size = inputs.size(0)

            # Forward pass (only regression output needed)
            reg_out, _ = model(inputs)

            # Score only the valid positions
            reg_out_scored = reg_out[:, : library.config.Config.SEQ_SCORED, :]
            targets_scored = targets[:, : library.config.Config.SEQ_SCORED, :]

            # Filter for scored columns ONLY (reactivity, deg_Mg_pH10, deg_Mg_50C)
            reg_out_scored = reg_out_scored[
                :, :, library.config.Config.SCORED_COLS_INDICES
            ]
            targets_scored = targets_scored[
                :, :, library.config.Config.SCORED_COLS_INDICES
            ]

            # Calculate metric
            loss = mcrmse_loss(reg_out_scored, targets_scored)

            running_mcrmse += loss.item() * batch_size
            dataset_size += batch_size

    avg_mcrmse = running_mcrmse / dataset_size
    return avg_mcrmse


def run_training(train_loader, val_loader):
    """
    Main training loop with Early Stopping and Model Checkpointing.
    """
    seed_everything(library.config.Config.SEED)
    device = torch.device(library.config.Config.DEVICE)
    print(f"Using device: {device}")

    # Initialize Model
    model = MaskedBiGRU()
    model.to(device)

    # Initialize Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=library.config.Config.LEARNING_RATE,
        weight_decay=library.config.Config.WEIGHT_DECAY,
    )

    # Initialize Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=library.config.Config.EPOCHS,
        eta_min=library.config.Config.SCHEDULER_MIN_LR,
    )

    # Training Loop Variables
    best_mcrmse = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(library.config.Config.EPOCHS):
        # Train
        train_loss, train_reg, train_recon = train_fn(
            model, train_loader, optimizer, device
        )

        # Validate
        val_mcrmse = eval_fn(model, val_loader, device)

        # Step Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Print Metrics
        print(
            f"Epoch {epoch+1}/{library.config.Config.EPOCHS} | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.6f} (Reg: {train_reg:.6f}, Recon: {train_recon:.6f}) | "
            f"Val MCRMSE: {val_mcrmse}"
        )

        # Checkpointing
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            patience_counter = 0
            torch.save(model.state_dict(), library.config.Config.MODEL_SAVE_PATH)
            print(f"New best model saved with MCRMSE: {best_mcrmse}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= library.config.Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Val MCRMSE: {best_mcrmse}")
    return best_mcrmse
