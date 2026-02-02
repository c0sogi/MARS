import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import set_seed, MCRMSELoss, compute_val_metric
from library.data import get_loaders
from library.model import RNAModel


def train_fn(model, loader, criterion, optimizer, device, scheduler=None):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (x, pair_indices, pair_mask, y) in enumerate(loader):
        x = x.to(device)
        pair_indices = pair_indices.to(device)
        pair_mask = pair_mask.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = model(x, pair_indices, pair_mask)

        # Loss calculation (MCRMSE on all 5 targets)
        loss = criterion(preds, y)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

        # Optimization step
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        dataset_size += x.size(0)

    # Note: Scheduler step is typically called per epoch in this setup,
    # but if using OneCycleLR it would be here. We use CosineAnnealingLR per epoch in run_training.

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def eval_fn(model, loader, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, pair_indices, pair_mask, y in loader:
            x = x.to(device)
            pair_indices = pair_indices.to(device)
            pair_mask = pair_mask.to(device)

            # Forward pass
            preds = model(x, pair_indices, pair_mask)

            # Move to CPU and collect
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute metric using the utility that handles slicing and column filtering
    val_score = compute_val_metric(all_preds, all_targets)

    return val_score


def inference_fn(model, loader, device):
    """
    Generates predictions for the test set.
    Returns:
        preds: np.array of shape (N, 107, 5)
        ids: list of sample IDs
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for x, pair_indices, pair_mask, sample_ids in loader:
            x = x.to(device)
            pair_indices = pair_indices.to(device)
            pair_mask = pair_mask.to(device)

            # Forward pass
            preds = model(x, pair_indices, pair_mask)

            all_preds.append(preds.cpu().numpy())
            all_ids.extend(sample_ids)

    all_preds = np.concatenate(all_preds, axis=0)
    return all_preds, all_ids


def run_training():
    """
    Main function to run the training pipeline, evaluation, and submission generation.
    """
    # 1. Setup
    set_seed(Config.seed)
    device = torch.device(Config.device)

    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_loaders(
        batch_size=Config.batch_size,
        num_workers=Config.num_workers,
        load_cached_data=True,
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = RNAModel(config=Config)
    model.to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_max, eta_min=Config.eta_min
    )

    criterion = MCRMSELoss()

    # 5. Training Loop
    best_score = float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.epochs):
        # Train
        train_loss = train_fn(model, train_loader, criterion, optimizer, device)

        # Validate
        val_score = eval_fn(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        # Logging
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{Config.epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCRMSE: {val_score} | "  # Full precision as requested
            f"LR: {current_lr:.2e}"
        )

        # Early Stopping & Model Saving
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  New best model saved! Score: {best_score}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{Config.patience}")

        if patience_counter >= Config.patience:
            print("Early stopping triggered.")
            break

    # 6. Submission Generation
    print("\nGenerating submission...")

    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    # Inference
    test_preds, test_ids = inference_fn(model, test_loader, device)

    # Format Submission
    # test_preds shape: (N_samples, 107, 5)
    # We need to flatten this to (N_samples * 107, 6) where columns are id_seqpos + 5 targets

    submission_data = []
    target_cols = (
        Config.target_cols
    )  # ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(test_ids):
        sample_preds = test_preds[i]  # (107, 5)
        for seqpos in range(Config.seq_length):
            row_id = f"{sample_id}_{seqpos}"
            row_preds = sample_preds[seqpos]

            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = float(row_preds[col_idx])

            submission_data.append(row_dict)

    submission_df = pd.DataFrame(submission_data)

    # Ensure column order matches sample submission
    cols = ["id_seqpos"] + target_cols
    submission_df = submission_df[cols]

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
