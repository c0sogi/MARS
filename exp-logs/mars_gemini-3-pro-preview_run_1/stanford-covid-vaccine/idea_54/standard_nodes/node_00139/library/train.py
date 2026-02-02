import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.data import load_data, RNADataset
from library.model import RNAModel
from library.utils import format_submission


def train_one_epoch(model, loader, optimizer, criterion, device, scheduler):
    """
    Trains the model for one epoch.
    Computes Masked MSE loss (first 68 positions), performs backpropagation,
    applies gradient clipping, and updates weights.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in loader:
        seq = batch["sequence"].to(device)
        loop = batch["loop"].to(device)
        dist = batch["distance"].to(device)
        target = batch["target"].to(device)  # Shape: (B, 68, 3)

        optimizer.zero_grad()

        # Forward pass
        # Output shape: (B, 107, 3)
        pred = model(seq, loop, dist)

        # Slice prediction to scored region (first 68 positions)
        pred_scored = pred[:, : Config.SEQ_SCORED, :]

        # Compute Loss (MSE)
        loss = criterion(pred_scored, target)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_NORM)

        # Optimizer Step
        optimizer.step()

        # Scheduler Step (Cosine Annealing per step)
        if scheduler:
            scheduler.step()

        running_loss += loss.item() * seq.size(0)
        dataset_size += seq.size(0)

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using column-wise MCRMSE.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            seq = batch["sequence"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["distance"].to(device)
            target = batch["target"].to(device)

            # Forward pass
            pred = model(seq, loop, dist)

            # Slice to scored region
            pred_scored = pred[:, : Config.SEQ_SCORED, :]

            all_preds.append(pred_scored.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    if not all_preds:
        return 0.0

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate MCRMSE column-wise
    # Shape: (N, 68, 3)
    # Mean over samples (axis 0) and sequence positions (axis 1) -> (3,)
    mse_per_col = np.mean((all_targets - all_preds) ** 2, axis=(0, 1))
    rmse_per_col = np.sqrt(mse_per_col)
    mcrmse_score = np.mean(rmse_per_col)

    return mcrmse_score


def run_training(epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, debug=False):
    """
    Orchestrates the training, validation, and submission generation pipeline.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    # load_data handles caching internally
    train_ids, train_seq, train_loop, train_dist, train_tgt = load_data("train")
    val_ids, val_seq, val_loop, val_dist, val_tgt = load_data("val")

    if debug:
        subset_size = 100
        train_seq = train_seq[:subset_size]
        train_loop = train_loop[:subset_size]
        train_dist = train_dist[:subset_size]
        train_tgt = train_tgt[:subset_size]
        val_seq = val_seq[:subset_size]
        val_loop = val_loop[:subset_size]
        val_dist = val_dist[:subset_size]
        val_tgt = val_tgt[:subset_size]
        epochs = 2

    # Create Datasets and Loaders
    train_dataset = RNADataset(train_seq, train_loop, train_dist, train_tgt)
    val_dataset = RNADataset(val_seq, val_loop, val_dist, val_tgt)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Initialize Model
    model = RNAModel(Config).to(device)

    # Optimizer (AdamW with low weight decay)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler (Cosine Annealing)
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Loss Function
    criterion = nn.MSELoss()

    # Training Loop
    best_mcrmse = float("inf")
    save_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    # Ensure cache dir exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scheduler
        )
        val_mcrmse = validate(model, val_loader, device)

        # Print full precision
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.10f} | Val MCRMSE: {val_mcrmse:.10f}"
        )

        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), save_path)

    print(f"Best Val MCRMSE: {best_mcrmse:.10f}")

    # Inference on Test Set
    print("Generating submission...")

    # Load Test Data
    test_ids, test_seq, test_loop, test_dist = load_data("test")

    # Create Test Loader
    test_dataset = RNADataset(test_seq, test_loop, test_dist, None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Load Best Model
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            seq = batch["sequence"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["distance"].to(device)

            # Forward pass
            pred = model(seq, loop, dist)  # (B, 107, 3)
            all_preds.append(pred.cpu().numpy())

    if all_preds:
        all_preds = np.concatenate(all_preds, axis=0)

        # Format and Save Submission
        format_submission(test_ids, all_preds, Config.SUBMISSION_PATH)
    else:
        print("No predictions generated.")
