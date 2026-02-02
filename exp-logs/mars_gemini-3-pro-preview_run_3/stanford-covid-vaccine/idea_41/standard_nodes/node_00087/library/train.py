import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.data import load_data, collate_fn
from library.model import DeepDecoupledBiGRU
from library.utils import seed_everything, mcrmse_loss


def training_loss_fn(y_pred, y_true):
    """
    Calculates MCRMSE loss on all 5 columns for training purposes.
    Slices inputs to the valid ground truth length (Config.PRED_LEN = 68).

    Args:
        y_pred: (Batch, Seq_Len, 5)
        y_true: (Batch, Seq_Len, 5)

    Returns:
        Scalar MCRMSE loss over all 5 columns.
    """
    # Slice to scored length (68)
    y_pred_sliced = y_pred[:, : Config.PRED_LEN, :]
    y_true_sliced = y_true[:, : Config.PRED_LEN, :]

    # Calculate MSE per column (averaging over Batch and Sequence dimensions)
    mse = torch.mean((y_true_sliced - y_pred_sliced) ** 2, dim=(0, 1))

    # Calculate RMSE per column
    rmse = torch.sqrt(mse)

    # Average RMSE across all 5 columns
    loss = torch.mean(rmse)

    return loss


def train_one_epoch(model, loader, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        features = batch["features"].to(device)
        pair_indices = batch["pair_indices"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = model(features, pair_indices)

        # Calculate loss on all 5 targets
        loss = training_loss_fn(preds, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping (Stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD_NORM)

        # Optimizer step
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using the official metric logic.
    Aggregates all predictions first to ensure correct global metric calculation.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            targets = batch["targets"].to(device)

            preds = model(features, pair_indices)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    # Concatenate to form full dataset tensors
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate official metric (handles slicing and column filtering internally)
    score = mcrmse_loss(all_targets, all_preds)

    return score.item()


def run_training():
    """
    Main training routine.
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Device: {device}")

    # 1. Load Data
    print("Loading datasets...")
    train_dataset = load_data("train", load_cached_data=True, debug=Config.DEBUG)
    val_dataset = load_data("val", load_cached_data=True, debug=Config.DEBUG)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # 2. Initialize Model
    print("Initializing Deep Decoupled BiGRU model...")
    model = DeepDecoupledBiGRU().to(device)

    # 3. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # 4. Training Loop
    best_score = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.MAX_EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validate
        val_score = validate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()

        # Logging
        print(
            f"Epoch {epoch+1}/{Config.MAX_EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score}"
        )

        # Checkpointing & Early Stopping
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  [Saved] New best model. Score: {best_score}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training finished. Best Validation MCRMSE: {best_score}")


def generate_submission():
    """
    Generates the submission file using the best trained model.
    """
    device = Config.DEVICE
    print("Generating submission...")

    # 1. Load Test Data
    test_dataset = load_data("test", load_cached_data=True, debug=Config.DEBUG)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # 2. Load Model
    model = DeepDecoupledBiGRU().to(device)
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
        print(f"Loaded model from {Config.MODEL_PATH}")
    else:
        print("Warning: Best model file not found. Using initialized weights.")

    model.eval()

    # 3. Inference
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            ids = batch["id"]

            # Predict
            preds = model(features, pair_indices)  # Shape: (B, 107, 5)

            all_preds.append(preds.cpu().numpy())
            all_ids.extend(ids)

    all_preds = np.concatenate(all_preds, axis=0)  # Shape: (N_samples, 107, 5)

    # 4. Format Submission
    # Flatten predictions to match sample_submission format: one row per seqpos
    submission_rows = []
    target_cols = Config.TARGET_COLS

    for i, sample_id in enumerate(all_ids):
        sample_pred = all_preds[i]  # (107, 5)

        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_values = sample_pred[seqpos]

            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = row_values[col_idx]

            submission_rows.append(row_dict)

    submission_df = pd.DataFrame(submission_rows)

    # Save
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
