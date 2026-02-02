import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from library.config import Config
from library.dataset import get_dataset, RNADataset
from library.model import RNAModel
from library.loss import MaskedMSELoss
from library.utils import seed_everything, calculate_mcrmse


def train_fn(model, loader, optimizer, criterion, device, grad_clip):
    """
    Executes one training epoch.

    Args:
        model: The PyTorch model.
        loader: DataLoader for training data.
        optimizer: The optimizer.
        criterion: The loss function.
        device: 'cuda' or 'cpu'.
        grad_clip: Gradient clipping value.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    total_loss = 0.0

    for x_seq, x_loop, x_dist, y in loader:
        x_seq = x_seq.to(device)
        x_loop = x_loop.to(device)
        x_dist = x_dist.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = model(x_seq, x_loop, x_dist)

        # Loss calculation
        # MaskedMSELoss handles slicing predictions to match target length internally
        loss = criterion(preds, y)

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def eval_fn(model, loader, device, pred_len):
    """
    Executes validation inference and calculates MCRMSE.

    Args:
        model: The PyTorch model.
        loader: DataLoader for validation data.
        device: 'cuda' or 'cpu'.
        pred_len: Length of the scored sequence (e.g., 68).

    Returns:
        float: The calculated MCRMSE score.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_seq, x_loop, x_dist, y in loader:
            x_seq = x_seq.to(device)
            x_loop = x_loop.to(device)
            x_dist = x_dist.to(device)

            preds = model(x_seq, x_loop, x_dist)

            # Slice predictions to match scored length (68) for metric calculation
            # preds shape: (B, 107, 3) -> (B, 68, 3)
            preds_scored = preds[:, :pred_len, :]

            all_preds.append(preds_scored.cpu())
            all_targets.append(y)

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate metric using the utility function
    score = calculate_mcrmse(all_targets, all_preds)
    return score


def run_training(
    debug=False, epochs=None, batch_size=None, patience=5, load_cached_data=True
):
    """
    Main training loop with Early Stopping.

    Args:
        debug (bool): If True, runs on a subset of data for few epochs.
        epochs (int): Number of epochs to train. Defaults to Config.EPOCHS.
        batch_size (int): Batch size. Defaults to Config.BATCH_SIZE.
        patience (int): Early stopping patience.
        load_cached_data (bool): Whether to load data from cache if available.

    Returns:
        str: Path to the best saved model.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # Configuration overrides
    num_epochs = epochs if epochs is not None else Config.EPOCHS
    b_size = batch_size if batch_size is not None else Config.BATCH_SIZE

    if debug:
        print("Debug mode enabled: reducing epochs and dataset size.")
        num_epochs = 2

    # 2. Data Loading
    # Uses library.dataset.get_dataset which handles caching
    train_ids, train_seq, train_loop, train_dist, train_y = get_dataset(
        Config.TRAIN_PATH, "train", load_cached_data=load_cached_data
    )
    val_ids, val_seq, val_loop, val_dist, val_y = get_dataset(
        Config.VAL_PATH, "val", load_cached_data=load_cached_data
    )

    # Debug subsetting
    if debug:
        subset_size = 100
        train_seq = train_seq[:subset_size]
        train_loop = train_loop[:subset_size]
        train_dist = train_dist[:subset_size]
        train_y = train_y[:subset_size]

        val_seq = val_seq[:subset_size]
        val_loop = val_loop[:subset_size]
        val_dist = val_dist[:subset_size]
        val_y = val_y[:subset_size]

    # Create Datasets and Loaders
    train_ds = RNADataset(train_seq, train_loop, train_dist, train_y)
    val_ds = RNADataset(val_seq, val_loop, val_dist, val_y)

    train_loader = DataLoader(
        train_ds,
        batch_size=b_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=b_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = RNAModel(Config).to(device)

    # 4. Optimization
    optimizer = AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = MaskedMSELoss(scored_len=Config.PRED_LEN)

    # 5. Training Loop
    best_mcrmse = float("inf")
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    epochs_no_improve = 0

    print(f"Starting training on {device} for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        # Train
        train_loss = train_fn(
            model, train_loader, optimizer, criterion, device, Config.GRAD_CLIP
        )

        # Validate
        val_mcrmse = eval_fn(model, val_loader, device, Config.PRED_LEN)

        # Scheduler Step
        scheduler.step()

        # Logging (Full precision)
        print(
            f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss} | Val MCRMSE: {val_mcrmse}"
        )

        # Checkpointing & Early Stopping
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with MCRMSE: {best_mcrmse}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(
                f"Early stopping triggered after {patience} epochs without improvement."
            )
            break

    print(f"Training finished. Best Validation MCRMSE: {best_mcrmse}")
    return best_model_path
