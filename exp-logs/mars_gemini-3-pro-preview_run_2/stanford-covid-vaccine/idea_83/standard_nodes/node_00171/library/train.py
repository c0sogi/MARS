import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import Config
from library.utils import set_seed, compute_global_rmse
from library.loss import MaskedMCRMSE
from library.data import get_loader
from library.model import HC_HIGFN


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, partner_indices, targets) in enumerate(loader):
        inputs = inputs.to(device)
        partner_indices = partner_indices.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass returns (final_pred, aux_pred)
        y_final, y_aux = model(inputs, partner_indices)

        # Calculate loss
        # Main loss on final prediction
        loss_final = criterion(y_final, targets)
        # Auxiliary loss on intermediate prediction
        loss_aux = criterion(y_aux, targets)

        # Weighted sum
        loss = loss_final + 0.5 * loss_aux

        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    return avg_loss


def validate(model, loader, device):
    """
    Runs validation and computes Global MCRMSE.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, partner_indices, targets in loader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)
            # targets are kept on CPU to save GPU memory during accumulation if large
            # but compute_global_rmse handles tensors, so we can keep them or move to CPU.
            # Moving to CPU for accumulation is safer for memory.

            # Forward pass (returns only final_pred in eval mode)
            y_final = model(inputs, partner_indices)

            all_preds.append(y_final.cpu())
            all_targets.append(targets.cpu())

    # Concatenate all batches
    y_pred_global = torch.cat(all_preds, dim=0)
    y_true_global = torch.cat(all_targets, dim=0)

    # Compute Global RMSE
    mcrmse, metrics = compute_global_rmse(y_pred_global, y_true_global)

    return mcrmse, metrics


def generate_submission(model, device):
    """
    Generates submission file for the test set.
    """
    print("Generating submission...")

    # Load test loader (shuffle=False is crucial to match IDs)
    test_loader = get_loader("test", batch_size=Config.BATCH_SIZE, shuffle=False)

    model.eval()
    all_preds = []

    with torch.no_grad():
        for inputs, partner_indices, _ in test_loader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)

            y_final = model(inputs, partner_indices)
            all_preds.append(y_final.cpu().numpy())

    # Shape: (N_samples, Seq_Len, 5)
    preds_array = np.concatenate(all_preds, axis=0)

    # Get IDs from the dataset
    ids = test_loader.dataset.ids

    # Prepare data for DataFrame
    # We need to flatten: id_seqpos, val1, val2, val3, val4, val5
    submission_data = []

    seq_len = preds_array.shape[1]
    target_cols = (
        Config.ALL_TARGET_COLS
    )  # ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Efficient flattening
    # Create ID column: id_0, id_1, ...
    # We can vectorize this construction

    print("Formatting predictions...")

    # Reshape predictions to (N_samples * Seq_Len, 5)
    flat_preds = preds_array.reshape(-1, 5)

    # Generate ID list
    # Repeat each ID seq_len times
    repeated_ids = np.repeat(ids, seq_len)

    # Generate sequence positions 0..106 repeated N_samples times
    tiled_positions = np.tile(np.arange(seq_len), len(ids))

    # Create id_seqpos strings
    # Using list comprehension as it's often faster for string manip than numpy char ops
    id_seqpos = [f"{i}_{p}" for i, p in zip(repeated_ids, tiled_positions)]

    # Create DataFrame
    sub_df = pd.DataFrame(flat_preds, columns=target_cols)
    sub_df.insert(0, "id_seqpos", id_seqpos)

    # Save
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_training(debug=False):
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Data Loaders
    print("Initializing Data Loaders...")
    train_loader = get_loader("train", batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = get_loader("val", batch_size=Config.BATCH_SIZE, shuffle=False)

    # 2. Model Setup
    print("Initializing Model...")
    model = HC_HIGFN().to(device)

    # Cite Lesson 00136: Do not mask sequence in training loss to anchor BiGRU on zero-padded tail
    criterion = MaskedMCRMSE(mask_sequence=False)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # 3. Training Loop
    best_mcrmse = float("inf")
    patience_counter = 0

    epochs = 2 if debug else Config.EPOCHS

    print("Starting Training...")
    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, Config.GRAD_CLIP
        )

        # Validate
        val_mcrmse, val_metrics = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step(val_mcrmse)

        elapsed = time.time() - start_time

        # Logging
        print(f"Epoch {epoch+1}/{epochs} | Time: {elapsed:.1f}s")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val MCRMSE: {val_mcrmse}")  # Full precision
        print(f"  Val Metrics: {val_metrics}")

        # Checkpointing & Early Stopping
        if val_mcrmse < best_mcrmse:
            print(f"  [New Best] MCRMSE improved from {best_mcrmse} to {val_mcrmse}")
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  [Patience] {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best MCRMSE: {best_mcrmse}")

    # 4. Submission
    print("Loading best model for submission...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    generate_submission(model, device)


if __name__ == "__main__":
    # This block is not required by the prompt instructions but facilitates local testing if run directly
    pass
