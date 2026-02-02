import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os

# Import from provided library files
from library.config import Config
from library.utils import set_seed, mcrmse_loss, get_scored_metrics
from library.dataset import get_loader
from library.model import RNAModel


def train_epoch(model, loader, optimizer, device):
    """
    Trains the model for one epoch using the Zero-Masked Non-Linear Channel-Gated BiGRU strategy.

    Args:
        model (nn.Module): The RNA model.
        loader (DataLoader): Training data loader.
        optimizer (Optimizer): The optimizer.
        device (torch.device): Compute device.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        inputs = batch["sequence"].to(device)
        adj = batch["adjacency"].to(device)
        mask = batch["mask"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()

        # Forward pass
        # Inputs: (B, 107, 14), Adj: (B, 107), Mask: (B, 107)
        preds = model(inputs, adj, mask)

        # Slice predictions to match scored length (68)
        # Targets are (B, 68, 5), Preds are (B, 107, 5)
        preds_sliced = preds[:, : Config.SEQ_SCORED, :]

        # Calculate Loss
        loss = mcrmse_loss(targets, preds_sliced)

        # Backward pass
        loss.backward()

        # Gradient Clipping (Mandatory for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    return avg_loss


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using Global Metric Aggregation.
    Concatenates all predictions before calculating metrics to avoid batch-averaging bias.

    Args:
        model (nn.Module): The RNA model.
        loader (DataLoader): Validation data loader.
        device (torch.device): Compute device.

    Returns:
        tuple: (val_loss, val_score)
    """
    model.eval()
    val_preds_list = []
    val_targets_list = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["sequence"].to(device)
            adj = batch["adjacency"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["target"].to(device)

            preds = model(inputs, adj, mask)

            # Slice predictions to scored length
            preds_sliced = preds[:, : Config.SEQ_SCORED, :]

            val_preds_list.append(preds_sliced.cpu())
            val_targets_list.append(targets.cpu())

    # Concatenate all batches for global metric calculation
    val_preds = torch.cat(val_preds_list, dim=0)
    val_targets = torch.cat(val_targets_list, dim=0)

    # Calculate metrics
    # MCRMSE on all 5 columns
    val_loss = mcrmse_loss(val_targets, val_preds).item()
    # MCRMSE on the 3 scored columns (reactivity, deg_Mg_pH10, deg_Mg_50C)
    val_score = get_scored_metrics(val_targets, val_preds)

    return val_loss, val_score


def run_training():
    """
    Main training loop with Early Stopping and Checkpointing.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # Load Data
    print("Initializing DataLoaders...")
    train_loader = get_loader("train", shuffle=True)
    val_loader = get_loader("val", shuffle=False)

    # Initialize Model
    print("Initializing Model...")
    model = RNAModel(Config).to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.T_MAX)

    best_score = float("inf")
    patience = 5  # Early stopping patience
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        # Train Step
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # Step Scheduler
        scheduler.step()

        # Validation Step
        val_loss, val_score = validate(model, val_loader, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val Score: {val_score}"
        )

        # Checkpointing & Early Stopping Logic
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  New best model saved! Score: {best_score}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"Training complete. Best Val Score: {best_score}")


def generate_submission():
    """
    Generates submission file using the best trained model.
    Predicts on the test set and saves to submission.csv.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Load Test Data
    print("Loading Test Data...")
    test_loader = get_loader("test", shuffle=False)

    # Load Model
    print(f"Loading model from {Config.MODEL_PATH}...")
    model = RNAModel(Config).to(device)
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    else:
        print("Error: Model checkpoint not found! Cannot generate submission.")
        return

    model.eval()
    results = []

    print("Running inference on test set...")
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["sequence"].to(device)
            adj = batch["adjacency"].to(device)
            mask = batch["mask"].to(device)
            ids = batch["id"]

            preds = model(inputs, adj, mask)  # (B, 107, 5)
            preds = preds.cpu().numpy()

            # Format predictions for submission
            # We need to predict for all 107 positions
            for i, sample_id in enumerate(ids):
                sample_preds = preds[i]  # (107, 5)

                for seqpos in range(Config.SEQ_LEN):
                    row_id = f"{sample_id}_{seqpos}"
                    row_preds = sample_preds[seqpos]

                    # Columns: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
                    # Order must match sample_submission.csv
                    results.append([row_id] + row_preds.tolist())

    # Save Submission
    cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    df_sub = pd.DataFrame(results, columns=cols)
    df_sub.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
