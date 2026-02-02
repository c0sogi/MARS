import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import seed_everything, MetricTracker
from library.loss import MCRMSELoss
from library.data import get_loaders
from library.model import TISRNModel


def train_one_epoch(model, loader, criterion, optimizer, device, config):
    """
    Trains the model for one epoch using the TI-SRN recycling strategy.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Unpack batch
        inputs = batch["inputs"].to(device)
        partner_indices = batch["partner_indices"].to(device)
        targets = batch["targets"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()

        # Forward pass returns tuple: (Predictions_Pass1, Predictions_Pass2)
        # Pass 1 uses zero feedback. Pass 2 uses detached Pass 1 as feedback.
        preds_pass1, preds_pass2 = model(inputs, partner_indices, mask)

        # Calculate loss for both passes
        # Note: criterion handles column slicing (reactivity, deg_Mg_pH10, deg_Mg_50C)
        loss_pass2 = criterion(preds_pass2, targets, mask)
        loss_pass1 = criterion(preds_pass1, targets, mask)

        # Weighted total loss
        loss = loss_pass2 + (config.aux_loss_weight * loss_pass1)

        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Uses only the final refined output (Pass 2) for metrics.
    """
    model.eval()
    tracker = MetricTracker()
    running_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            partner_indices = batch["partner_indices"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)

            # Forward pass
            _, preds_pass2 = model(inputs, partner_indices, mask)

            # Compute loss for monitoring (Pass 2 only)
            loss = criterion(preds_pass2, targets, mask)
            running_loss += loss.item() * inputs.size(0)

            # Update metric tracker
            # Mask predictions/targets before updating tracker to ensure clean scoring
            # (Though tracker handles shapes, explicit masking is safer for custom metrics)
            if mask is not None:
                # Expand mask: (B, L) -> (B, L, 1)
                mask_expanded = mask.unsqueeze(-1).cpu().numpy()
                preds_np = preds_pass2.cpu().numpy() * mask_expanded
                targets_np = targets.cpu().numpy() * mask_expanded

                # Convert back to tensor for tracker compatibility or modify tracker usage
                # The provided MetricTracker takes tensors, detaches them, and computes.
                # We can pass the masked tensors.
                tracker.update(torch.tensor(preds_np), torch.tensor(targets_np))
            else:
                tracker.update(preds_pass2, targets)

    epoch_loss = running_loss / len(loader.dataset)
    mcrmse = tracker.compute()

    return epoch_loss, mcrmse


def run_training():
    """
    Main execution function for training the TI-SRN model.
    """
    # 1. Configuration & Setup
    config = Config()
    seed_everything(config.seed)

    # Ensure working directory exists
    os.makedirs(config.cache_dir, exist_ok=True)

    print(f"Device: {config.device}")
    print(f"Cache Directory: {config.cache_dir}")

    # 2. Data Loaders
    print("Initializing Data Loaders...")
    train_loader, val_loader, test_loader = get_loaders(config, load_cached_data=True)

    # 3. Model Initialization
    print("Initializing TI-SRN Model...")
    model = TISRNModel(config).to(config.device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        verbose=True,
    )

    criterion = MCRMSELoss()

    # 5. Training Loop
    best_mcrmse = float("inf")
    early_stop_counter = 0
    best_model_path = os.path.join(config.cache_dir, "best_model.pth")

    print(f"Starting training for {config.epochs} epochs...")

    for epoch in range(config.epochs):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, config.device, config
        )

        # Validate
        val_loss, val_mcrmse = validate(model, val_loader, criterion, config.device)

        # Scheduler Step
        scheduler.step(val_mcrmse)

        # Logging
        print(
            f"Epoch {epoch+1}/{config.epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val MCRMSE: {val_mcrmse}"
        )

        # Checkpointing & Early Stopping
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            early_stop_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved! MCRMSE: {best_mcrmse}")
        else:
            early_stop_counter += 1
            print(
                f"  No improvement. Early stopping counter: {early_stop_counter}/{config.early_stopping_patience}"
            )

        if early_stop_counter >= config.early_stopping_patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation MCRMSE: {best_mcrmse}")

    # 6. Generate Submission (Optional but recommended to verify pipeline)
    # We load the best model and run inference on test set if needed,
    # but the prompt specifically asks to implement the module logic.
    # We will ensure the best model is saved as required.


if __name__ == "__main__":
    run_training()
