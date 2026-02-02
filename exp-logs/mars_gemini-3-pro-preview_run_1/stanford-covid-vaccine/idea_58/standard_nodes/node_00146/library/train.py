import os
import torch
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import seed_everything, calculate_mcrmse
from library.data import get_dataloaders
from library.model import RNAModel, masked_mse_loss, predict_and_submit


def train_epoch(model, loader, optimizer, device, grad_clip):
    """
    Executes one training epoch.

    Args:
        model: The neural network model.
        loader: DataLoader for training data.
        optimizer: The optimizer instance.
        device: 'cuda' or 'cpu'.
        grad_clip: Max norm for gradient clipping.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for seq, loop, dist, targets, mask in loader:
        seq = seq.to(device)
        loop = loop.to(device)
        dist = dist.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()

        preds = model(seq, loop, dist)
        loss = masked_mse_loss(preds, targets, mask)

        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using MCRMSE.

    Args:
        model: The neural network model.
        loader: DataLoader for validation data.
        device: 'cuda' or 'cpu'.

    Returns:
        float: The calculated MCRMSE score.
    """
    model.eval()
    val_preds_list = []
    val_targets_list = []

    with torch.no_grad():
        for seq, loop, dist, targets, mask in loader:
            seq = seq.to(device)
            loop = loop.to(device)
            dist = dist.to(device)
            targets = targets.to(device)

            preds = model(seq, loop, dist)

            # Slice to scored positions (first 68) for metric calculation
            pred_len = Config.PRED_LEN
            preds_sliced = preds[:, :pred_len, :]
            targets_sliced = targets[:, :pred_len, :]

            val_preds_list.append(preds_sliced.cpu().numpy())
            val_targets_list.append(targets_sliced.cpu().numpy())

    val_preds = np.concatenate(val_preds_list, axis=0)
    val_targets = np.concatenate(val_targets_list, axis=0)

    mcrmse = calculate_mcrmse(val_targets, val_preds)
    return mcrmse


def run_training(epochs=Config.EPOCHS, debug=Config.DEBUG, patience=5):
    """
    Main execution function for training, validation, and submission.

    Args:
        epochs (int): Number of training epochs.
        debug (bool): Whether to run in debug mode (subset of data).
        patience (int): Early stopping patience.
    """
    # Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # Configure Debug Mode
    Config.DEBUG = debug

    # Load Data
    # Note: get_dataloaders handles caching internally based on Config.DEBUG
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Initialize Model
    model = RNAModel().to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training Loop State
    best_mcrmse = float("inf")
    best_model_path = os.path.join(Config.SUBMISSION_DIR, "best_model.pth")
    patience_counter = 0

    print(f"Starting training for {epochs} epochs on {device}...")
    print(
        f"Model: Vector-Scaled High-Capacity Wide-Stream BiGRU (Dim: {Config.HIDDEN_DIM})"
    )

    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device, Config.GRAD_CLIP
        )

        # Step Scheduler
        scheduler.step()

        # Validate
        val_mcrmse = validate(model, val_loader, device)

        # Logging (Full precision)
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val MCRMSE: {val_mcrmse}"
        )

        # Checkpoint & Early Stopping
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training complete. Best Val MCRMSE: {best_mcrmse}")

    # Load Best Model for Submission
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Generate Submission
    predict_and_submit(model, test_loader)
