import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import set_seed, mcrmse
from library.data import get_dataloaders
from library.model import InteractionAwareModel


class MaskedMSELoss(nn.Module):
    """
    Computes MSE loss only on the scored positions (first 68) and scored targets.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, preds, targets):
        """
        Args:
            preds: (Batch, Seq_Len=107, 3)
            targets: (Batch, Pred_Len=68, 3)
        """
        # Slice predictions to match the scored length
        preds_sliced = preds[:, : Config.PRED_LEN, :]
        return self.mse(preds_sliced, targets)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        # Move data to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        targets = targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimizer step
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / num_batches
    return avg_loss


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using MCRMSE.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass
            outputs = model(inputs)

            # Slice predictions to scored length (Batch, 68, 3)
            outputs_sliced = outputs[:, : Config.PRED_LEN, :]

            # Store results on CPU
            all_preds.append(outputs_sliced.cpu().numpy())
            all_targets.append(targets.numpy())

    # Concatenate all batches
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    # Calculate MCRMSE
    score = mcrmse(y_true, y_pred)
    return score


def run_training():
    """
    Main function to run the training pipeline.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data
    print("Initializing DataLoaders...")
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=True)

    # 3. Model
    print("Initializing Model...")
    model = InteractionAwareModel().to(device)

    # 4. Optimization
    criterion = MaskedMSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN)

    # 5. Training Loop
    best_score = float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_score = validate(model, val_loader, device)

        # Scheduler step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        elapsed = time.time() - start_time

        # Logging
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Time: {elapsed:.2f}s | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss} | "
            f"Val MCRMSE: {val_score}"
        )

        # Early Stopping & Model Saving
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  >>> New Best Model Saved! Score: {best_score}")
        else:
            patience_counter += 1
            print(
                f"  >>> No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
            )

            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Validation MCRMSE: {best_score}")
