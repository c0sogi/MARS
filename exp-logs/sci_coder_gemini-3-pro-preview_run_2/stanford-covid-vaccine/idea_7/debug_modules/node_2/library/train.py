import os
import time
import torch
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import set_seed, MetricTracker
from library.loss import MaskedMCRMSELoss
from library.data import get_dataloaders
from library.model import HybridNet


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.

    Args:
        model: The neural network model.
        loader: DataLoader for the training set.
        optimizer: The optimizer.
        criterion: The loss function.
        device: The device to run on (CPU or CUDA).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for batch in loader:
        # Move inputs to device
        inputs = batch["inputs"].to(device)
        partner_indices = batch["partner_indices"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = model(inputs, partner_indices)

        # Calculate loss
        loss = criterion(preds, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Accumulate loss (weighted by batch size for accurate mean)
        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        count += batch_size

    epoch_loss = running_loss / count if count > 0 else 0.0
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The neural network model.
        loader: DataLoader for the validation set.
        criterion: The loss function (for monitoring purposes).
        device: The device to run on.

    Returns:
        tuple: (MCRMSE Score, Average Loss)
    """
    model.eval()
    running_loss = 0.0
    count = 0

    tracker = MetricTracker()

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            partner_indices = batch["partner_indices"].to(device)
            targets = batch["targets"].to(device)

            # Forward pass
            preds = model(inputs, partner_indices)

            # Calculate loss for monitoring
            loss = criterion(preds, targets)
            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            count += batch_size

            # Update metric tracker (handles slicing and masking internally)
            tracker.update(preds, targets)

    avg_loss = running_loss / count if count > 0 else 0.0
    mcrmse_score = tracker.result()

    return mcrmse_score, avg_loss


def run_training(epochs=Config.EPOCHS, load_cached_data=True):
    """
    Orchestrates the training process including data loading, model initialization,
    training loop, validation, scheduler stepping, and early stopping.

    Args:
        epochs (int): Maximum number of epochs to train.
        load_cached_data (bool): Whether to use cached data files.

    Returns:
        model: The trained model with the best validation weights loaded.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=load_cached_data)

    # 3. Model Initialization
    print("Initializing Model...")
    model = HybridNet().to(device)

    # 4. Loss, Optimizer, Scheduler
    criterion = MaskedMCRMSELoss().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    # 5. Training Loop
    best_mcrmse = float("inf")
    epochs_no_improve = 0
    best_model_path = Config.MODEL_CHECKPOINT

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_mcrmse, val_loss = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        # Print metrics (Full precision as requested)
        print(
            f"Epoch {epoch}/{epochs} | Time: {elapsed:.2f}s | "
            f"Train Loss: {train_loss} | Val Loss: {val_loss} | "
            f"Val MCRMSE: {val_mcrmse}"
        )

        # Scheduler Step
        scheduler.step(val_mcrmse)

        # Checkpointing and Early Stopping
        if val_mcrmse < best_mcrmse:
            print(
                f"Validation MCRMSE improved from {best_mcrmse} to {val_mcrmse}. Saving model..."
            )
            best_mcrmse = val_mcrmse
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    # 6. Load Best Model
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: No model checkpoint found. Returning current model.")

    return model
