import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from library.config import Config
from library.utils import get_device


class WeightedL1Loss(nn.Module):
    """
    Custom L1 Loss that weights inspiratory and expiratory phases differently.
    """

    def __init__(self):
        super().__init__()
        self.w_insp = Config.LOSS_INSPIRATORY_WEIGHT
        self.w_exp = Config.LOSS_EXPIRATORY_WEIGHT

    def forward(self, pred, target, u_out):
        """
        Args:
            pred: (Batch, Seq_Len, 1)
            target: (Batch, Seq_Len)
            u_out: (Batch, Seq_Len) - 0 for insp, 1 for exp
        """
        pred = pred.squeeze(-1)
        diff = torch.abs(pred - target)

        # Apply weights: w_insp where u_out=0, w_exp where u_out=1
        weights = (1 - u_out) * self.w_insp + u_out * self.w_exp

        loss = (diff * weights).mean()
        return loss


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0.0

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        u_out = batch["u_out"].to(device)

        optimizer.zero_grad()

        # Forward
        preds = model(x)

        # Loss
        loss = criterion(preds, y, u_out)

        # Backward
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Step
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on validation set.
    Returns:
        avg_loss: Weighted L1 Loss
        avg_mae: MAE on Inspiratory phase (u_out=0)
    """
    model.eval()
    total_loss = 0.0
    total_mae_sum = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            u_out = batch["u_out"].to(device)

            preds = model(x)

            # Loss
            loss = criterion(preds, y, u_out)
            total_loss += loss.item()

            # Metric: MAE on Inspiratory phase only
            preds_flat = preds.squeeze(-1)
            diff = torch.abs(preds_flat - y)

            mask = u_out == 0
            masked_diff = diff[mask]

            total_mae_sum += masked_diff.sum().item()
            total_count += mask.sum().item()

    avg_loss = total_loss / len(loader)
    avg_mae = total_mae_sum / total_count if total_count > 0 else 0.0

    return avg_loss, avg_mae


def predict(model, loader, device):
    """
    Generates raw predictions for the dataset.
    Returns:
        np.array: Flattened predictions
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            preds = model(x)
            all_preds.append(preds.cpu().numpy())

    # Concatenate: (N_batches, Batch, 80, 1) -> (Total_Breaths, 80, 1)
    if len(all_preds) > 0:
        predictions = np.concatenate(all_preds, axis=0)
    else:
        return np.array([])

    # Flatten: (Total_Breaths * 80)
    return predictions.flatten()


def fit(model, train_loader, val_loader, optimizer, scheduler, device, epochs):
    """
    Main training loop with Early Stopping.
    """
    criterion = WeightedL1Loss()
    best_mae = float("inf")
    patience_counter = 0

    print(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae = evaluate(model, val_loader, criterion, device)

        # Scheduler Step (Cosine Annealing is per epoch)
        if scheduler is not None:
            scheduler.step()

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val MAE (Insp): {val_mae}"
        )

        # Early Stopping & Checkpointing
        if val_mae < best_mae:
            best_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Training complete. Best Val MAE: {best_mae}")


def generate_submission(model, test_loader, device):
    """
    Generates predictions for test set and saves submission.csv.
    """
    print("Generating predictions for test set...")
    preds = predict(model, test_loader, device)

    # Load Test Metadata to map predictions to IDs
    # Note: The model processes data sorted by breath_id, then time_step.
    # We must ensure metadata is sorted identically to assign preds correctly.
    # Since time_step is not in test_metadata, we sort by breath_id and id,
    # relying on the fact that id increases with time_step within a breath.
    if not os.path.exists(Config.TEST_METADATA):
        raise FileNotFoundError(f"Test metadata not found at {Config.TEST_METADATA}")

    df_meta = pd.read_csv(Config.TEST_METADATA)

    # Sort by breath_id, then id to match the data loading order
    df_meta = df_meta.sort_values(by=[Config.BREATH_ID_COL, "id"])

    # Verify lengths match
    if len(preds) != len(df_meta):
        raise ValueError(
            f"Prediction length {len(preds)} does not match metadata length {len(df_meta)}"
        )

    df_meta["pressure"] = preds

    # Create submission dataframe: id, pressure
    # Sort by id as per sample submission requirement
    submission = df_meta[["id", "pressure"]].sort_values(by="id")

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
