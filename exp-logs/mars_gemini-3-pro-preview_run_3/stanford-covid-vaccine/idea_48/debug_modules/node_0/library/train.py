import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import seed_everything, MCRMSELoss, compute_mcrmse_numpy
from library.data import get_dataloaders
from library.model import RNAModel


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Move data to device
        inputs = batch["inputs"].to(device)
        pair_indices = batch["pair_indices"].to(device)
        pair_masks = batch["pair_masks"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs, pair_indices, pair_masks)

        # Compute loss
        # MCRMSELoss handles slicing to pred_len (68) and filtering scored columns internally
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping (Mandatory for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.gradient_clip)

        # Update weights
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using global aggregation.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(inputs, pair_indices, pair_masks)

            # Collect full sequence outputs (CPU numpy)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate to (N_total, SeqLen, NumClasses)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute Metric
    # compute_mcrmse_numpy handles slicing to pred_len (68) and filtering scored columns
    score = compute_mcrmse_numpy(all_preds, all_targets)

    return score


def inference(model, loader, device):
    """
    Generates predictions for the test set.
    Returns:
        preds: (N, SeqLen, NumClasses) numpy array
        ids: List of IDs
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            ids = batch["id"]

            outputs = model(inputs, pair_indices, pair_masks)

            all_preds.append(outputs.cpu().numpy())
            all_ids.extend(ids)

    all_preds = np.concatenate(all_preds, axis=0)
    return all_preds, all_ids


def generate_submission(preds, ids, output_path):
    """
    Formats predictions into the submission CSV format.

    Args:
        preds: (N, 107, 5) numpy array
        ids: List of sample IDs
        output_path: Path to save the CSV
    """
    # Columns as per sample submission
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    submission_data = []

    for i, sample_id in enumerate(ids):
        sample_preds = preds[i]  # (107, 5)

        for seqpos in range(Config.seq_len):
            row_id = f"{sample_id}_{seqpos}"
            row_values = sample_preds[seqpos].tolist()

            # Create row dict
            row_dict = {"id_seqpos": row_id}
            for col_name, val in zip(target_cols, row_values):
                row_dict[col_name] = val

            submission_data.append(row_dict)

    df_sub = pd.DataFrame(submission_data)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(debug_samples=None, load_cached_data=True):
    """
    Main driver function for training and inference.
    """
    # 1. Setup
    seed_everything(Config.seed)
    Config.setup()
    device = torch.device(Config.device)

    print(f"Device: {device}")
    print(
        f"Debug Mode: {'ON (' + str(debug_samples) + ' samples)' if debug_samples else 'OFF'}"
    )

    # 2. Data Loading
    train_loader, val_loader, test_loader = get_dataloaders(
        debug_samples=debug_samples, load_cached_data=load_cached_data
    )

    # 3. Model Initialization
    model = RNAModel().to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=Config.epochs)
    criterion = MCRMSELoss()

    # 5. Training Loop
    best_score = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_score = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCRMSE: {val_score:.10f} | "
            f"Time: {elapsed:.2f}s"
        )

        # Early Stopping & Checkpointing
        if val_score < (best_score - Config.min_delta):
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.model_save_path)
            print(f"  -> New best model saved! Score: {best_score:.10f}")
        else:
            patience_counter += 1
            print(
                f"  -> No improvement. Patience: {patience_counter}/{Config.patience}"
            )

        if patience_counter >= Config.patience:
            print("Early stopping triggered.")
            break

    # 6. Final Inference
    print("\nLoading best model for inference...")
    model.load_state_dict(torch.load(Config.model_save_path, map_location=device))

    print("Generating test predictions...")
    test_preds, test_ids = inference(model, test_loader, device)

    print(f"Test predictions shape: {test_preds.shape}")

    # 7. Generate Submission
    generate_submission(test_preds, test_ids, Config.submission_file)

    print("Done.")
