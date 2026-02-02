import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import from provided library files
from library.config import Config
from library.data_utils import get_dataloaders
from library.model import DDCGBiGRU
from library.loss_metric import MCRMSELoss, competition_metric


def set_seed(seed=Config.SEED):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        loader: DataLoader for training data.
        optimizer: The optimizer.
        criterion: The loss function.
        device: The computing device.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Move data to device
        features = batch["features"].to(device)
        adj_indices = batch["adj_indices"].to(device)
        pair_mask = batch["pair_mask"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(features, adj_indices, pair_mask)

        # Calculate loss (MCRMSELoss handles slicing internally)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping (Critical for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP)

        # Update weights
        optimizer.step()

        running_loss += loss.item() * features.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Aggregates all predictions and targets to compute the global MCRMSE.

    Args:
        model: The PyTorch model.
        loader: DataLoader for validation data.
        device: The computing device.

    Returns:
        float: The competition metric (MCRMSE on scored columns).
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            adj_indices = batch["adj_indices"].to(device)
            pair_mask = batch["pair_mask"].to(device)
            targets = batch["targets"]  # Keep targets on CPU for final metric calc

            outputs = model(features, adj_indices, pair_mask)

            # Move outputs to CPU and collect
            all_preds.append(outputs.cpu())
            all_targets.append(targets)

    # Concatenate all batches
    global_preds = torch.cat(all_preds, dim=0)
    global_targets = torch.cat(all_targets, dim=0)

    # Calculate metric using the official competition metric function
    score = competition_metric(global_preds, global_targets)

    return score


def train_model(debug=False):
    """
    Main training loop with Early Stopping and Scheduler.

    Args:
        debug (bool): If True, runs on a subset of data.
    """
    set_seed()
    device = torch.device(Config.DEVICE)
    print(f"Training on device: {device}")

    # Data Loaders
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
        debug=debug,
    )

    # Model Setup
    model = DDCGBiGRU().to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    criterion = MCRMSELoss()

    # Tracking
    best_score = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_score = validate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()

        # Logging
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score}"
        )

        # Early Stopping & Checkpointing
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"New best model saved with MCRMSE: {best_score}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"Training complete. Best Validation Score: {best_score}")


def generate_submission(debug=False):
    """
    Generates the submission file using the best trained model.
    """
    set_seed()
    device = torch.device(Config.DEVICE)

    # Load Test Data
    _, _, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
        debug=debug,
    )

    # Load Model
    model = DDCGBiGRU().to(device)
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model file not found at {Config.MODEL_SAVE_PATH}. Train the model first."
        )

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    print("Generating predictions...")

    ids_list = []
    preds_list = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            adj_indices = batch["adj_indices"].to(device)
            pair_mask = batch["pair_mask"].to(device)
            batch_ids = batch["id"]

            # Forward pass: (B, 107, 5)
            outputs = model(features, adj_indices, pair_mask)

            preds_list.append(outputs.cpu().numpy())
            ids_list.extend(batch_ids)

    # Concatenate predictions: (N_samples, 107, 5)
    all_preds = np.concatenate(preds_list, axis=0)

    # Prepare submission data
    # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    submission_data = []

    for i, sample_id in enumerate(ids_list):
        sample_preds = all_preds[i]  # (107, 5)

        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_preds = sample_preds[seqpos]

            # Order matches Config.TARGET_COLS:
            # reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
            submission_data.append(
                [
                    row_id,
                    row_preds[0],
                    row_preds[1],
                    row_preds[2],
                    row_preds[3],
                    row_preds[4],
                ]
            )

    # Create DataFrame
    columns = ["id_seqpos"] + Config.TARGET_COLS
    submission_df = pd.DataFrame(submission_data, columns=columns)

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
