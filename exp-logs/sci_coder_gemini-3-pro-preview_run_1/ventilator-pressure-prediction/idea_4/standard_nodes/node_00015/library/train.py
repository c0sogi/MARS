import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything
from library.dataset import prepare_data
from library.model import PhysicsResidualNet


def masked_mae_loss(y_pred, y_true, u_out):
    """
    Calculates Mean Absolute Error (MAE) only for the inspiratory phase (u_out == 0).

    Args:
        y_pred: Predicted pressure (Batch, Seq_Len)
        y_true: Actual pressure (Batch, Seq_Len)
        u_out: Control input for expiratory valve (Batch, Seq_Len)

    Returns:
        Scalar loss value.
    """
    # Create mask: 1 where u_out is 0 (inspiratory), 0 otherwise
    mask = 1 - u_out

    # Calculate absolute error
    mae = torch.abs(y_pred - y_true)

    # Apply mask to ignore expiratory phase
    masked_mae = mae * mask

    # Calculate mean over the valid elements
    # Add epsilon to denominator to avoid division by zero
    loss = masked_mae.sum() / (mask.sum() + 1e-8)

    return loss


def train_one_epoch(model, loader, optimizer, scheduler, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        # Move data to device
        x = batch["x"].to(device)
        u_out = batch["u_out"].to(device)
        y = batch["y"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        preds = model(x)

        # Calculate loss
        loss = masked_mae_loss(preds, y, u_out)

        # Backward pass
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

        # Optimizer and Scheduler steps
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def valid_one_epoch(model, loader, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            u_out = batch["u_out"].to(device)
            y = batch["y"].to(device)

            preds = model(x)
            loss = masked_mae_loss(preds, y, u_out)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def run_training():
    """
    Main function to run the training pipeline.
    """
    # Set seed for reproducibility
    seed_everything(Config.seed)

    print(f"Starting training for experiment: {Config.exp_name}")
    print(f"Device: {Config.device}")

    # Load Datasets
    print("Preparing datasets...")
    # Ensure working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    train_dataset = prepare_data("train", load_cached_data=True)
    val_dataset = prepare_data("val", load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.val_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Initialize Model
    model = PhysicsResidualNet()
    model.to(Config.device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    # Scheduler (OneCycleLR)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=Config.epochs,
        pct_start=Config.pct_start,
        div_factor=Config.div_factor,
        final_div_factor=Config.final_div_factor,
    )

    # Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training loop...")
    for epoch in range(Config.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, Config.device
        )
        val_loss = valid_one_epoch(model, val_loader, Config.device)

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{Config.epochs} - Train Loss: {train_loss} - Val Loss: {val_loss}"
        )

        # Checkpoint and Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.model_path)
            print(f"New best model saved with Val Loss: {best_val_loss}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.patience}")

        if patience_counter >= Config.patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Loss: {best_val_loss}")


def predict_and_submit():
    """
    Loads the best model, generates predictions on the test set, and saves the submission file.
    """
    print("Starting inference...")
    seed_everything(Config.seed)

    # Load Test Data
    test_dataset = prepare_data("test", load_cached_data=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.val_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Load Model
    model = PhysicsResidualNet()
    model.to(Config.device)

    if not os.path.exists(Config.model_path):
        raise FileNotFoundError(
            f"Model file not found at {Config.model_path}. Run training first."
        )

    print(f"Loading model from {Config.model_path}...")
    model.load_state_dict(torch.load(Config.model_path, map_location=Config.device))
    model.eval()

    all_preds = []
    all_ids = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(Config.device)
            ids = batch["ids"].numpy()  # IDs are needed for submission mapping

            # Predict
            preds = model(x)

            # Move to CPU and flatten
            # The model outputs (Batch, Seq_Len), we need to flatten to (Batch * Seq_Len)
            preds_np = preds.cpu().numpy().flatten()
            ids_np = ids.flatten()

            all_preds.append(preds_np)
            all_ids.append(ids_np)

    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_ids = np.concatenate(all_ids)

    # Create Submission DataFrame
    submission_df = pd.DataFrame({Config.id_col: all_ids, Config.target_col: all_preds})

    # Ensure ID is int
    submission_df[Config.id_col] = submission_df[Config.id_col].astype(int)

    # Save
    print(f"Saving submission to {Config.submission_path}...")
    submission_df.to_csv(Config.submission_path, index=False)
    print("Submission saved successfully.")
