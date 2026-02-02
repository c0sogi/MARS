import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from library.config import Config
from library.data import get_dataloaders, get_test_dataloader
from library.model import RNAModel
from library.utils import seed_everything, mcrmse_loss, format_submission


def train_one_epoch(model, loader, optimizer, device):
    """
    Trains the model for one epoch using Masked MSE loss and Gradient Clipping.
    """
    model.train()
    running_loss = 0.0

    for seq, loop, dist, tgt in loader:
        seq = seq.to(device)
        loop = loop.to(device)
        dist = dist.to(device)
        tgt = tgt.to(device)

        optimizer.zero_grad()

        # Forward pass
        pred = model(seq, loop, dist)

        # Masked MSE Loss: Only calculate loss for the first 68 positions (Config.PRED_LEN)
        # pred: (B, 107, 3), tgt: (B, 107, 3) -> slice to (B, 68, 3)
        loss = F.mse_loss(pred[:, : Config.PRED_LEN, :], tgt[:, : Config.PRED_LEN, :])

        loss.backward()

        # Gradient Clipping to stabilize 512-width BiLSTM
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using the MCRMSE metric.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for seq, loop, dist, tgt in loader:
            seq = seq.to(device)
            loop = loop.to(device)
            dist = dist.to(device)

            pred = model(seq, loop, dist)

            all_preds.append(pred.cpu())
            all_targets.append(tgt.cpu())

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate MCRMSE (Mean Columnwise Root Mean Squared Error)
    # This utility function handles the slicing and column-wise averaging
    score = mcrmse_loss(all_preds, all_targets)

    return score


def train_model(epochs=Config.EPOCHS, max_samples=None, patience=5, save_path=None):
    """
    Main training pipeline with Early Stopping.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Setup directories
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    if save_path is None:
        save_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    # Load Data
    print(f"Loading data (max_samples={max_samples})...")
    train_loader, val_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
        max_samples=max_samples,
    )

    # Initialize Model
    print("Initializing Orthogonally-Initialized High-Capacity BiLSTM...")
    model = RNAModel().to(device)

    # Optimizer & Scheduler
    # Low weight decay (1e-4) to avoid suppressing recurrent signals
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_mcrmse = float("inf")
    patience_counter = 0

    print(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_mcrmse = validate(model, val_loader, device)

        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1}/{epochs} | LR: {current_lr:.2e} | Train MSE: {train_loss:.6f} | Val MCRMSE: {val_mcrmse}"
        )

        # Early Stopping & Model Checkpointing
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            # print(f"  New best model saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Training complete. Best Val MCRMSE: {best_mcrmse}")
    return save_path


def generate_submission_file(model_path, max_samples=None):
    """
    Generates predictions for the test set and saves the submission file.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print("Loading test data...")
    test_loader, test_ids = get_test_dataloader(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
        max_samples=max_samples,
    )

    print(f"Loading model from {model_path}...")
    model = RNAModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []

    print("Running inference...")
    with torch.no_grad():
        for seq, loop, dist in test_loader:
            seq = seq.to(device)
            loop = loop.to(device)
            dist = dist.to(device)

            # Forward pass
            pred = model(seq, loop, dist)  # (B, 107, 3)
            all_preds.append(pred.cpu())

    all_preds = torch.cat(all_preds, dim=0)

    print("Formatting and saving submission...")
    format_submission(test_ids, all_preds, save_dir=Config.SUBMISSION_DIR)


def run_training(epochs=Config.EPOCHS, max_samples=None):
    """
    Wrapper function to run the full training and submission pipeline.
    """
    best_model_path = train_model(epochs=epochs, max_samples=max_samples)
    generate_submission_file(best_model_path, max_samples=max_samples)
    return best_model_path
