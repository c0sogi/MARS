import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from library.config import Config
from library.model import MSDHNet
from library.data_utils import get_data_loaders


def set_seed(seed):
    """
    Sets the random seed for reproducibility across numpy, torch, and python random.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def masked_mae_loss(y_pred, y_true, u_out):
    """
    Calculates Mean Absolute Error (MAE) loss only for the inspiratory phase.
    The inspiratory phase is defined where u_out == 0.
    """
    # Create mask where u_out == 0 (inspiratory phase)
    mask = (u_out == 0).float()

    # Calculate absolute error
    loss = torch.abs(y_pred - y_true)

    # Apply mask to the loss
    masked_loss = loss * mask

    # Calculate mean over the valid elements (sum of mask)
    # Add a small epsilon to avoid division by zero in case a batch has no inspiratory phase (unlikely)
    return masked_loss.sum() / (mask.sum() + 1e-8)


def train_one_epoch(model, loader, optimizer, device):
    """
    Runs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        # Extract u_out for masking.
        # According to Config.FEATURE_COLS, 'u_out' is at index 1.
        # x shape: (Batch, Seq_Len, Features)
        u_out = x[:, :, 1]

        optimizer.zero_grad()

        # Forward pass
        preds = model(x)

        # Calculate masked loss
        loss = masked_mae_loss(preds, y, u_out)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def evaluate(model, loader, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            # Extract u_out for masking
            u_out = x[:, :, 1]

            preds = model(x)
            loss = masked_mae_loss(preds, y, u_out)
            running_loss += loss.item()

    return running_loss / len(loader)


def predict(model, loader, device):
    """
    Generates predictions for the test set.
    Returns a flattened numpy array of predictions matching the submission format.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            preds = model(x)

            # Model outputs (Batch, Seq_Len).
            # We need to flatten this to match the row-wise submission ID structure.
            preds_flat = preds.view(-1).cpu().numpy()
            all_preds.append(preds_flat)

    return np.concatenate(all_preds)


def run_training(load_cached_data=True, debug=False):
    """
    Main function to execute the training pipeline.

    Args:
        load_cached_data (bool): Whether to load pre-processed .npy files.
        debug (bool): Whether to run in debug mode with a data subset.

    Returns:
        float: The best validation loss achieved.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    # get_data_loaders handles caching and feature engineering
    print("Preparing DataLoaders...")
    train_loader, val_loader, test_loader, test_ids = get_data_loaders(
        load_cached_data=load_cached_data, debug=debug
    )

    # 3. Model Initialization
    print("Initializing MSDH-Net...")
    model = MSDHNet().to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: Reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        verbose=True,
        min_lr=Config.MIN_LR,
    )

    # 5. Training Loop
    best_val_loss = float("inf")
    early_stop_counter = 0
    early_stop_patience = 15  # Stop if no improvement for 15 epochs

    epochs = Config.EPOCHS
    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)

        # Step the scheduler based on validation loss
        scheduler.step(val_loss)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.10f} | Val Loss: {val_loss:.10f}"
        )

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            early_stop_counter = 0
            print(f"  -> New best model saved! Loss: {best_val_loss:.10f}")
        else:
            early_stop_counter += 1

        # Early Stopping
        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # 6. Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))

    print("Generating predictions...")
    predictions = predict(model, test_loader, device)

    # 7. Submission
    # Ensure predictions match test_ids length.
    # In standard operation, these should match exactly.
    if len(predictions) != len(test_ids):
        print(
            f"Warning: Prediction length {len(predictions)} != ID length {len(test_ids)}"
        )
        min_len = min(len(predictions), len(test_ids))
        predictions = predictions[:min_len]
        test_ids = test_ids[:min_len]

    submission_df = pd.DataFrame(
        {Config.ID_COL: test_ids, Config.TARGET_COL: predictions}
    )

    # Ensure directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    return best_val_loss
