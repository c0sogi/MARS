import time
import torch
import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.data import load_data
from library.model import RCRDN, generate_submission
from library.loss import MCRMSELoss


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Executes one training epoch.
    Handles the recycling output (list of preds) via the weighted MCRMSELoss.
    """
    model.train()
    total_loss = 0.0

    for x, p_idx, p_mask, y in loader:
        x = x.to(device)
        p_idx = p_idx.to(device)
        p_mask = p_mask.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # Forward pass returns list of predictions (recycling steps)
        preds_list = model(x, p_idx, p_mask)

        # Criterion calculates weighted loss over all recycling steps
        loss = criterion(preds_list, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Metrics are calculated based on the FINAL prediction of the recycling loop.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x, p_idx, p_mask, y in loader:
            x = x.to(device)
            p_idx = p_idx.to(device)
            p_mask = p_mask.to(device)
            y = y.to(device)

            preds_list = model(x, p_idx, p_mask)

            # For validation metric, we strictly evaluate the final prediction
            final_pred = preds_list[-1]

            # Calculate MCRMSE on the final prediction
            loss = criterion(final_pred, y)
            total_loss += loss.item()

    return total_loss / len(loader)


def run_training(debug=False, epochs=None):
    """
    Main driver function for training, validation, and submission.
    """
    set_seed(42)

    # Allow overriding epochs for debugging/testing
    if epochs is None:
        epochs = Config.EPOCHS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading training data...")
    train_dataset = load_data("train", load_cached_data=True, debug=debug)
    print("Loading validation data...")
    val_dataset = load_data("val", load_cached_data=True, debug=debug)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # 2. Initialize Model
    model = RCRDN().to(device)

    # 3. Initialize Loss
    # Weights for recycling steps: [0.5, 1.0] implies the first pass counts for 33%
    # and the final pass for 66% of the gradient signal.
    criterion = MCRMSELoss(weights=[0.5, 1.0])

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")

    # 5. Training Loop
    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        # Scheduler step
        scheduler.step(val_loss)

        elapsed = time.time() - start_time

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Time: {elapsed:.2f}s"
        )

        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"New best model saved! Loss: {best_val_loss}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val Loss: {best_val_loss}")

    # 6. Submission
    if os.path.exists(Config.MODEL_PATH):
        print("Loading best model for submission...")
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

        # Generate Submission using the library function
        generate_submission(model, device)
    else:
        print("No model saved. Skipping submission.")
