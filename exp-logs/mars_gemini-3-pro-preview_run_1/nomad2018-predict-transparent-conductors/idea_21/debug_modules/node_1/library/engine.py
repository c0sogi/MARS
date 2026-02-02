import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.model import CRRD_DeepSets
from library.data import get_train_val_loaders
from library.utils import rmsle


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Unpack batch
        atomic_feats = batch["atomic_feats"].to(device)
        batch_indices = batch["batch_indices"].to(device)
        global_feats = batch["global_feats"].to(device)
        targets = batch["targets"].to(device)

        # Transform targets to log space: log(1 + y)
        # This aligns MSE loss with the RMSLE metric
        log_targets = torch.log1p(targets)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(atomic_feats, batch_indices, global_feats)

        # Compute loss
        loss = criterion(outputs, log_targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * targets.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss (MSE on log targets) and RMSLE (on original scale).
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            atomic_feats = batch["atomic_feats"].to(device)
            batch_indices = batch["batch_indices"].to(device)
            global_feats = batch["global_feats"].to(device)
            targets = batch["targets"].to(device)

            # Transform targets for loss calculation
            log_targets = torch.log1p(targets)

            # Forward pass (outputs are in log space)
            outputs = model(atomic_feats, batch_indices, global_feats)

            # Loss calculation
            loss = criterion(outputs, log_targets)
            running_loss += loss.item() * targets.size(0)

            # Inverse transform outputs for RMSLE calculation: exp(x) - 1
            # Clip outputs to avoid overflow/underflow before exp
            outputs = torch.clamp(outputs, min=-20, max=20)
            preds_linear = torch.expm1(outputs)

            all_preds.append(preds_linear.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    # Concatenate all batches
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # Calculate RMSLE using the utility function
    # Note: rmsle utility applies log1p internally, so we pass linear scale values
    epoch_rmsle = rmsle(all_targets, all_preds)

    return epoch_loss, epoch_rmsle


def fit(
    load_cached_data=True,
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
):
    """
    Main training loop with Early Stopping and Scheduler.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading data...")
    train_loader, val_loader = get_train_val_loaders(
        load_cached_data=load_cached_data, batch_size=batch_size
    )

    # 2. Initialize Model
    model = CRRD_DeepSets().to(device)

    # 3. Setup Optimization
    # Using AdamW with weight decay as specified in strategy
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: Reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Loss function: MSE
    criterion = nn.MSELoss()

    # Early Stopping variables
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = Config.BEST_MODEL_PATH

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_rmsle = evaluate(model, val_loader, criterion, device)

        # Update Scheduler
        scheduler.step(val_loss)

        # Logging
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.8f} | "
            f"Val Loss: {val_loss:.8f} | "
            f"Val RMSLE: {val_rmsle:.8f}"
        )

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            # print(f"  New best model saved to {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"Training complete. Best Validation Loss: {best_val_loss:.8f}")

    # Load best model for return
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    return model
