import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm

from library.config import Config
from library.utils import seed_everything, GlobalMetricTracker
from library.loss import MaskedMCRMSELoss
from library.data import get_loaders
from library.model import ScalePartitionedDenseNet


def train_epoch(model, loader, optimizer, loss_fn, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    # We slice to SEQ_SCORED to ensure we only train on valid ground truth positions
    seq_scored = Config.SEQ_SCORED

    for inputs, partner_indices, targets in loader:
        inputs = inputs.to(device)
        partner_indices = partner_indices.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = model(inputs, partner_indices)

        # Slice to scored sequence length for loss calculation
        # Targets are zero-padded beyond seq_scored, so we strictly train on valid data
        preds_sliced = preds[:, :seq_scored, :]
        targets_sliced = targets[:, :seq_scored, :]

        # Compute loss (MaskedMCRMSELoss handles column selection internally)
        loss = loss_fn(preds_sliced, targets_sliced)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, loss_fn, device):
    """
    Performs validation and calculates the global MCRMSE metric.
    """
    model.eval()
    running_loss = 0.0
    tracker = GlobalMetricTracker()

    seq_scored = Config.SEQ_SCORED

    # Determine indices of scored columns for the metric tracker
    # Config.TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Config.SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    target_cols = Config.TARGET_COLS
    scored_cols = set(Config.SCORED_COLS)
    scored_indices = [i for i, col in enumerate(target_cols) if col in scored_cols]

    with torch.no_grad():
        for inputs, partner_indices, targets in loader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)
            targets = targets.to(device)

            preds = model(inputs, partner_indices)

            # Slice sequence length
            preds_sliced = preds[:, :seq_scored, :]
            targets_sliced = targets[:, :seq_scored, :]

            # 1. Compute Loss (Loss fn handles column masking internally)
            loss = loss_fn(preds_sliced, targets_sliced)
            running_loss += loss.item()

            # 2. Update Metric Tracker
            # Tracker computes error on all passed channels. We must manually select
            # the scored columns to get the correct MCRMSE.
            preds_metric = preds_sliced[:, :, scored_indices]
            targets_metric = targets_sliced[:, :, scored_indices]

            tracker.update(preds_metric, targets_metric)

    avg_loss = running_loss / len(loader)
    mcrmse = tracker.compute()

    return avg_loss, mcrmse


def inference(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for inputs, partner_indices, _ in loader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)

            # Forward pass
            preds = model(inputs, partner_indices)

            # Move to CPU and store
            all_preds.append(preds.cpu().numpy())

    # Concatenate all batches: Shape (N_samples, 107, 5)
    return np.concatenate(all_preds, axis=0)


def generate_submission(preds, ids, output_path):
    """
    Formats predictions into the submission CSV format.
    """
    # preds shape: (N_samples, 107, 5)
    # ids shape: (N_samples,)

    target_cols = Config.TARGET_COLS
    submission_rows = []

    num_samples, seq_len, num_targets = preds.shape

    # Efficiently construct data for DataFrame
    # We need to flatten: sample 0 pos 0, sample 0 pos 1, ... sample 1 pos 0 ...

    # 1. Create ID column: id_seqpos
    # Repeat IDs seq_len times
    ids_repeated = np.repeat(ids, seq_len)
    # Tile seq positions: 0, 1, ..., 106, 0, 1, ...
    seq_pos_tiled = np.tile(np.arange(seq_len), num_samples)

    id_seqpos = [f"{i}_{p}" for i, p in zip(ids_repeated, seq_pos_tiled)]

    # 2. Flatten predictions
    # Reshape to (N_samples * 107, 5)
    preds_flat = preds.reshape(-1, num_targets)

    # 3. Create DataFrame
    df_data = {"id_seqpos": id_seqpos}
    for i, col in enumerate(target_cols):
        df_data[col] = preds_flat[:, i]

    submission_df = pd.DataFrame(df_data)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(epochs=None, batch_size=None):
    """
    Main execution function for training and submission generation.
    """
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)

    # Override config if arguments provided
    if epochs is not None:
        Config.EPOCHS = epochs
    if batch_size is not None:
        Config.BATCH_SIZE = batch_size

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_loaders()

    # 3. Model
    print("Initializing model...")
    model = ScalePartitionedDenseNet().to(device)

    # 4. Optimization
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    loss_fn = MaskedMCRMSELoss().to(device)

    # 5. Training Loop
    best_mcrmse = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.IDEA_DIR, "best_model.pth")

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_mcrmse = validate(model, val_loader, loss_fn, device)

        # Step scheduler
        scheduler.step(val_mcrmse)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val MCRMSE: {val_mcrmse:.10f}"
        )

        # Checkpointing & Early Stopping
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            # print("  New best model saved.")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Best Val MCRMSE: {best_mcrmse:.10f}")

    # 6. Inference
    print("Generating predictions on test set...")
    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_preds = inference(model, test_loader, device)

    # 7. Submission
    print("Creating submission file...")
    test_ids = test_loader.dataset.ids
    generate_submission(test_preds, test_ids, Config.SUBMISSION_PATH)
