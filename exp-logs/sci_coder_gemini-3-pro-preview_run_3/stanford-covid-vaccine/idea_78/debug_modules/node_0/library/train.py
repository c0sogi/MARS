import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from library.config import Config
from library.dataset import RNADataset
from library.model import DeepResidualBiGRU
from library.loss import MCRMSELoss
from library.metrics import compute_scored_mcrmse


def set_seed(seed):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.

    Args:
        model: The PyTorch model.
        loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to run on (cpu or cuda).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_samples = 0

    for batch in loader:
        # Move data to device
        features = batch["features"].to(device)
        adjacency = batch["adjacency"].to(device)
        bpp_mask = batch["bpp_mask"].to(device)
        targets = batch["targets"].to(device)

        batch_size = features.size(0)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(features, adjacency, bpp_mask)

        # Loss calculation
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimizer step
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item() * batch_size
        num_samples += batch_size

    epoch_loss = running_loss / num_samples if num_samples > 0 else 0.0
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Runs inference on the validation set and computes metrics.

    Args:
        model: The PyTorch model.
        loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to run on.

    Returns:
        tuple: (average_loss, mcrmse_score)
    """
    model.eval()
    running_loss = 0.0
    num_samples = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            adjacency = batch["adjacency"].to(device)
            bpp_mask = batch["bpp_mask"].to(device)
            targets = batch["targets"].to(device)

            batch_size = features.size(0)

            # Forward pass
            outputs = model(features, adjacency, bpp_mask)

            # Loss calculation
            loss = criterion(outputs, targets)
            running_loss += loss.item() * batch_size
            num_samples += batch_size

            # Store predictions and targets for metric calculation
            # Move to CPU to save GPU memory
            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())

    avg_loss = running_loss / num_samples if num_samples > 0 else 0.0

    # Concatenate all batches
    if all_preds:
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Compute official metric
        mcrmse_score = compute_scored_mcrmse(all_preds, all_targets)
    else:
        mcrmse_score = 0.0

    return avg_loss, mcrmse_score


def run_training(epochs=None, debug=False):
    """
    Orchestrates the training process.

    Args:
        epochs (int, optional): Number of epochs to train. Defaults to Config.EPOCHS.
        debug (bool, optional): If True, uses a small subset of data for debugging.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    num_epochs = epochs if epochs is not None else Config.EPOCHS

    print(f"Running training on device: {device}")
    print(f"Epochs: {num_epochs}, Debug Mode: {debug}")

    # 2. Data Loading
    print("Loading datasets...")
    train_dataset = RNADataset(split="train")
    val_dataset = RNADataset(split="val")

    if debug:
        # Use a small subset for debugging
        subset_size = 64
        train_indices = list(range(min(len(train_dataset), subset_size)))
        val_indices = list(range(min(len(val_dataset), subset_size)))
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
        print(
            f"Debug mode: Reduced dataset sizes to {len(train_dataset)} train, {len(val_dataset)} val."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = DeepResidualBiGRU().to(device)

    # 4. Optimization Setup
    criterion = MCRMSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 5. Training Loop
    best_metric = float("inf")
    patience_counter = 0

    print("Starting training loop...")
    for epoch in range(num_epochs):
        start_time = os.times()[4]  # Monotonic time

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_metric = validate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step()

        # Logging
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"  Train Loss: {train_loss}")
        print(f"  Val Loss: {val_loss}")
        print(f"  Val Metric (MCRMSE): {val_metric}")

        # Checkpointing
        if val_metric < best_metric:
            print(
                f"  [Improvement] Metric improved from {best_metric} to {val_metric}. Saving model..."
            )
            best_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            print(f"  [No Improvement] Patience: {patience_counter}/{Config.PATIENCE}")

        # Early Stopping
        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training finished. Best Validation MCRMSE: {best_metric}")
