import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from library.config import Config
from library.dataset import load_data
from library.model import WideResBiGRU
from library.utils import mcrmse_loss


def set_seed(seed):
    """Sets the seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0.0
    loss_fn = nn.MSELoss()

    for batch in loader:
        # Move inputs to device
        seq = batch["sequence"].to(device)
        loop = batch["loop_type"].to(device)
        dist = batch["distance"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Forward pass
        # Output shape: (Batch, Seq_Len, 3)
        preds = model(seq, loop, dist)

        # Slice to scored length for loss calculation
        # We only train on the first 68 positions
        preds_scored = preds[:, : Config.PRED_LEN, :]
        targets_scored = targets[:, : Config.PRED_LEN, :]

        # Compute MSE Loss
        loss = loss_fn(preds_scored, targets_scored)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using MCRMSE.
    """
    model.eval()
    total_mcrmse = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            seq = batch["sequence"].to(device)
            loop = batch["loop_type"].to(device)
            dist = batch["distance"].to(device)
            targets = batch["targets"].to(device)

            # Forward pass
            preds = model(seq, loop, dist)

            # Calculate MCRMSE
            # The utility function handles slicing internally
            score = mcrmse_loss(targets, preds)

            # Accumulate
            # Note: Averaging batch MCRMSEs is an approximation of the global MCRMSE,
            # but standard for batch-wise validation monitoring.
            total_mcrmse += score.item()
            num_batches += 1

    return total_mcrmse / num_batches


def generate_submission(model, device):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("Generating submission...")

    # Load test data
    test_dataset = load_data(mode="test", load_cached_data=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    model.eval()

    # Storage for predictions
    all_ids = []
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            seq = batch["sequence"].to(device)
            loop = batch["loop_type"].to(device)
            dist = batch["distance"].to(device)
            ids = batch["id"]  # List of IDs

            # Forward pass
            preds = model(seq, loop, dist)

            all_preds.append(preds.cpu().numpy())
            all_ids.extend(ids)

    # Concatenate all predictions: (N_Samples, 107, 3)
    all_preds = np.concatenate(all_preds, axis=0)

    # Prepare submission data
    # We need to flatten the predictions to one row per (id, seqpos)
    submission_rows = []

    # Target columns predicted by the model
    pred_cols = Config.TARGET_COLS  # ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']

    # All columns required by submission
    req_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Map predicted columns to indices in the output array
    col_indices = {col: i for i, col in enumerate(pred_cols)}

    for i, sample_id in enumerate(all_ids):
        sample_preds = all_preds[i]  # (107, 3)

        for pos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{pos}"
            row_data = {"id_seqpos": row_id}

            for col in req_cols:
                if col in col_indices:
                    # Use model prediction
                    val = sample_preds[pos, col_indices[col]]
                else:
                    # Fill with 0 for ignored columns
                    val = 0.0
                row_data[col] = val

            submission_rows.append(row_data)

    # Create DataFrame
    df_sub = pd.DataFrame(submission_rows)

    # Save
    df_sub.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")


def train_model(debug=False, subset_size=None):
    """
    Main training pipeline.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading datasets...")
    train_dataset = load_data(
        mode="train", load_cached_data=True, subset_size=subset_size
    )
    val_dataset = load_data(mode="val", load_cached_data=True, subset_size=subset_size)

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

    # 2. Initialize Model
    print("Initializing model...")
    model = WideResBiGRU().to(device)

    # 3. Optimizer and Scheduler
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # 4. Training Loop
    best_mcrmse = float("inf")

    print(f"Starting training for {Config.EPOCHS} epochs...")
    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validate
        val_mcrmse = validate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCRMSE: {val_mcrmse:.10f} | "
            f"LR: {current_lr:.2e}"
        )

        # Checkpoint
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"  >>> New Best Model Saved (MCRMSE: {best_mcrmse:.10f})")

    print(f"Training complete. Best Validation MCRMSE: {best_mcrmse:.10f}")

    # 5. Generate Submission
    # Load best weights
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    generate_submission(model, device)


if __name__ == "__main__":
    # Can be run with subset_size for debugging if needed,
    # but default behavior is full training.
    train_model()
