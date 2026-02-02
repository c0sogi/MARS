import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import Config, seed_everything
from library.utils import get_device, compute_mae
from library.dataset import get_data_loaders
from library.model import VentilatorModel


def masked_l1_loss(preds, targets, u_out):
    """
    Computes L1 loss masked by u_out (only inspiratory phase, u_out == 0).
    """
    mask = (u_out == 0).float()
    loss = torch.abs(preds - targets)
    loss = (loss * mask).sum() / mask.sum()
    return loss


def train_fn(model, loader, optimizer, scheduler, device):
    """
    Training loop for one epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    # Progress bar for feedback (optional, but good for long epochs)
    # Using simple print logic to avoid clutter if tqdm is restricted,
    # but standard tqdm is usually fine. We'll stick to logic.

    for batch in loader:
        optimizer.zero_grad()

        # Move data to device
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        u_out = batch["u_out"].to(device)

        # Forward pass (returns tuple in training mode)
        final_pred, aux_pred = model(x)

        # Compute Losses
        loss_final = masked_l1_loss(final_pred, y, u_out)
        loss_aux = masked_l1_loss(aux_pred, y, u_out)

        # Composite Loss
        loss = loss_final + (Config.AUX_WEIGHT * loss_aux)

        # Backward
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Update
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def valid_fn(model, loader, device):
    """
    Validation loop for one epoch.
    """
    model.eval()
    total_mae = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            u_out = batch["u_out"].to(device)

            # Forward pass (returns only final_pred in eval mode)
            preds = model(x)

            # Compute Metric
            # compute_mae expects tensors or numpy arrays and handles the masking internally
            mae = compute_mae(preds, y, u_out)

            total_mae += mae
            num_batches += 1

    return total_mae / num_batches


def inference_fn(model, loader, device):
    """
    Generates predictions for the test set and saves submission.
    """
    model.eval()
    predictions = []
    ids_list = []

    print("Starting inference...")
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            ids = batch["ids"]  # Keep on CPU or move as needed, usually CPU for storage

            preds = model(x)

            # Flatten predictions and ids
            # preds shape: (batch, seq_len)
            # ids shape: (batch, seq_len)

            predictions.append(preds.cpu().numpy().flatten())
            ids_list.append(ids.numpy().flatten())

    # Concatenate all batches
    predictions = np.concatenate(predictions)
    ids_list = np.concatenate(ids_list)

    # Create DataFrame
    submission = pd.DataFrame({"id": ids_list, "pressure": predictions})

    # Sort by ID just in case (though loaders should preserve order)
    submission.sort_values(by="id", inplace=True)

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_training():
    """
    Main orchestration function.
    """
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Using device: {device}")

    # 1. Data Loaders
    train_loader, val_loader, test_loader = get_data_loaders(load_cached_data=True)

    # 2. Model
    model = VentilatorModel()
    model.to(device)

    # 3. Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # 4. Scheduler
    # OneCycleLR needs total steps
    total_steps = Config.EPOCHS * len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LR,
        total_steps=total_steps,
        pct_start=0.3,  # Standard usually, or adjusted based on convergence needs
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=10000.0,
    )

    # 5. Training Loop
    best_mae = float("inf")
    model_save_path = os.path.join(Config.WORKING_DIR, "model.pth")

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_fn(model, train_loader, optimizer, scheduler, device)
        val_mae = valid_fn(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss} | Val MAE: {val_mae}"
        )

        # Save Best Model
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved with MAE: {best_mae}")

    print(f"Training complete. Best Val MAE: {best_mae}")

    # 6. Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    inference_fn(model, test_loader, device)


if __name__ == "__main__":
    # This block is technically not required by the prompt instructions ("DO NOT include..."),
    # but standard for a script. I will leave the function definition available.
    # To execute: run_training()
    pass
