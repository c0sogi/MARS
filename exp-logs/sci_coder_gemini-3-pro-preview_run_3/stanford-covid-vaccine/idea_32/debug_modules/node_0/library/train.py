import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import Config
from library.utils import seed_everything, scored_mcrmse
from library.dataset import get_loaders
from library.model import SDIN_CG_BiGRU


def train_fn(model, loader, optimizer, criterion, device, scheduler):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        # Move data to device
        features = batch["features"].to(device)
        indices = batch["indices"].to(device)
        mask = batch["mask"].to(device)
        targets = batch["targets"].to(device)

        batch_size = features.size(0)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(features, indices, mask)

        # Slice to scored sequence length (68) for loss calculation
        # Ground truth is only provided for the first 68 bases.
        outputs_scored = outputs[:, : Config.SEQ_SCORED, :]
        targets_scored = targets[:, : Config.SEQ_SCORED, :]

        # Calculate Loss (MSE on all 5 columns)
        loss = criterion(outputs_scored, targets_scored)

        # Backward pass
        loss.backward()

        # Gradient Clipping (Critical for Deep RNNs)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimizer Step
        optimizer.step()

        # Scheduler Step (Cosine Annealing updates per step or per epoch?
        # Standard PyTorch CosineAnnealingLR is per epoch, but usually called per batch in some implementations.
        # Given T_MAX=EPOCHS in Config, it implies per-epoch stepping.
        # However, if we want per-step, T_MAX should be steps.
        # We will step scheduler outside the batch loop if it's epoch-based,
        # or check type. Standard CosineAnnealingLR is epoch-based.
        # We'll step it in the main loop.)

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def eval_fn(model, loader, device):
    """
    Evaluates the model on the validation set using the competition metric.
    """
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            features = batch["features"].to(device)
            indices = batch["indices"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["targets"].to(device)

            # Forward pass
            outputs = model(features, indices, mask)

            # Collect results (keep on CPU to save GPU memory)
            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate MCRMSE using the official utility
    # This utility handles slicing ([:68]) and column filtering internally.
    mcrmse = scored_mcrmse(all_preds, all_targets)

    return mcrmse.item()


def inference(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", leave=False):
            features = batch["features"].to(device)
            indices = batch["indices"].to(device)
            mask = batch["mask"].to(device)
            ids = batch["id"]

            outputs = model(features, indices, mask)

            all_preds.append(outputs.cpu().numpy())
            all_ids.extend(ids)

    all_preds = np.concatenate(all_preds, axis=0)
    return all_preds, all_ids


def run_training():
    """
    Main training management loop.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE

    print(f"Initializing training on {device}...")
    Config.print_config()

    # 2. Data Loading
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)
    print("Data loaders ready.")

    # 3. Model Initialization
    model = SDIN_CG_BiGRU().to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # Loss Function (MSE is a stable surrogate for MCRMSE training)
    criterion = nn.MSELoss()

    # 5. Training Loop
    best_mcrmse = float("inf")
    early_stop_counter = 0

    print("\nStarting training loop...")
    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_fn(
            model, train_loader, optimizer, criterion, device, scheduler
        )

        # Validate
        val_mcrmse = eval_fn(model, val_loader, device)

        # Step Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss (MSE): {train_loss:.6f} | "
            f"Val MCRMSE: {val_mcrmse} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {elapsed:.2f}s"
        )

        # Checkpointing & Early Stopping
        if val_mcrmse < best_mcrmse:
            print(f"  >>> Improved! Saving model to {Config.MODEL_SAVE_PATH}")
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(
                f"  >>> No improvement. Patience: {early_stop_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if early_stop_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"\nTraining finished. Best Validation MCRMSE: {best_mcrmse}")

    # 6. Submission Generation
    print("\nGenerating submission...")

    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    # Predict on Test Set
    preds, ids = inference(model, test_loader, device)
    # preds shape: (N_samples, 107, 5)

    # Flatten predictions for submission format
    # Format: id_seqpos, reactivity, deg_Mg_pH10, ...

    submission_rows = []
    target_cols = (
        Config.TARGET_COLS
    )  # ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

    for i, sample_id in enumerate(ids):
        sample_preds = preds[i]  # (107, 5)
        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_values = sample_preds[seqpos].tolist()

            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = row_values[col_idx]

            submission_rows.append(row_dict)

    submission_df = pd.DataFrame(submission_rows)

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Submission shape: {submission_df.shape}")


if __name__ == "__main__":
    run_training()
