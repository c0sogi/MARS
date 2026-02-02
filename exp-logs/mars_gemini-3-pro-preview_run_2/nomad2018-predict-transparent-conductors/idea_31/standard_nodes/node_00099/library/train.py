import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import Config
from library.data import get_dataloaders
from library.model import SRACGN
from library.utils import (
    set_seed,
    StandardScaler,
    compute_metrics,
    save_checkpoint,
    load_checkpoint,
)


def fit_and_cache_scaler(train_loader, cache_dir, load_cached=True):
    """
    Fits the StandardScaler on training data targets or loads a cached scaler.
    """
    scaler_path = os.path.join(cache_dir, "target_scaler.npz")
    scaler = StandardScaler()

    if load_cached and os.path.exists(scaler_path):
        try:
            scaler.load(scaler_path)
            return scaler
        except Exception as e:
            print(f"Failed to load cached scaler: {e}. Re-fitting.")

    print("Fitting scaler on training data...")
    all_targets = []
    for batch in train_loader:
        if batch.y is not None:
            all_targets.append(batch.y)

    if not all_targets:
        raise ValueError("No targets found in training loader.")

    all_targets = torch.cat(all_targets, dim=0)
    scaler.fit(all_targets)
    scaler.save(scaler_path)
    return scaler


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Normalize targets
        targets_norm = scaler.transform(batch.y)

        # Forward pass
        preds_norm = model(batch)

        # Compute loss
        loss = criterion(preds_norm, targets_norm)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate(model, loader, criterion, scaler, device):
    """
    Evaluates the model on the validation set.
    Returns average loss (normalized) and metrics (denormalized).
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    all_preds_raw = []
    all_targets_raw = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # Normalize targets for loss calculation
            targets_norm = scaler.transform(batch.y)

            # Forward pass
            preds_norm = model(batch)

            # Compute loss on normalized values
            loss = criterion(preds_norm, targets_norm)
            total_loss += loss.item()
            num_batches += 1

            # Denormalize for metrics
            preds_raw = scaler.inverse_transform(preds_norm)

            all_preds_raw.append(preds_raw.cpu())
            all_targets_raw.append(batch.y.cpu())

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    if all_preds_raw:
        all_preds_raw = torch.cat(all_preds_raw, dim=0)
        all_targets_raw = torch.cat(all_targets_raw, dim=0)
        metrics = compute_metrics(all_preds_raw, all_targets_raw)
    else:
        metrics = {}

    return avg_loss, metrics


def run_training(
    num_epochs=Config.NUM_EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    load_cached_data=True,
):
    """
    Main training loop.
    """
    set_seed(Config.SEED)
    device = Config.DEVICE

    # 1. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=batch_size, load_cached=load_cached_data
    )

    # 2. Scaler
    scaler = fit_and_cache_scaler(
        train_loader, Config.CACHE_DIR, load_cached=load_cached_data
    )

    # 3. Model Setup
    model = SRACGN(config=Config).to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.SCHEDULER_MIN_LR,
    )
    criterion = nn.MSELoss()

    # 4. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_metrics = evaluate(model, val_loader, criterion, scaler, device)

        # Scheduler step
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Logging
        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch}/{num_epochs} | Time: {epoch_time:.2f}s | LR: {current_lr:.2e}"
        )
        print(f"  Train Loss (MSE Norm): {train_loss:.6f}")
        print(f"  Val Loss (MSE Norm):   {val_loss:.6f}")
        print(f"  Val RMSLE:             {val_metrics.get('mean_rmsle', 0.0):.6f}")
        if "formation_energy_rmsle" in val_metrics:
            print(
                f"    Formation RMSLE:     {val_metrics['formation_energy_rmsle']:.6f}"
            )
        if "bandgap_energy_rmsle" in val_metrics:
            print(f"    Bandgap RMSLE:       {val_metrics['bandgap_energy_rmsle']:.6f}")

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_loss, scaler, best_model_path)
            print("  [Saved Best Model]")
        else:
            patience_counter += 1
            print(
                f"  [No Improvement] Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val Loss: {best_val_loss:.6f}")


def generate_submission(load_cached_data=True):
    """
    Generates submission file using the best trained model.
    """
    set_seed(Config.SEED)
    device = Config.DEVICE
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    if not os.path.exists(best_model_path):
        print(f"Error: Best model not found at {best_model_path}. Run training first.")
        return

    # 1. Load Data
    _, _, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached=load_cached_data
    )

    # 2. Load Model and Scaler
    model = SRACGN(config=Config).to(device)
    scaler = StandardScaler()

    # Load checkpoint
    checkpoint = load_checkpoint(model, None, best_model_path, scaler)
    if checkpoint is None:
        return

    model.eval()

    # 3. Inference
    ids = []
    formation_energies = []
    bandgap_energies = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)

            # Forward pass
            preds_norm = model(batch)

            # Denormalize
            preds_raw = scaler.inverse_transform(preds_norm)

            # Clamp negative values to 0 (physical constraint)
            preds_raw = torch.clamp(preds_raw, min=0.0)

            # Collect results
            batch_ids = batch.material_id.cpu().numpy().flatten()
            batch_preds = preds_raw.cpu().numpy()

            ids.extend(batch_ids)
            formation_energies.extend(batch_preds[:, 0])
            bandgap_energies.extend(batch_preds[:, 1])

    # 4. Save Submission
    df_sub = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": formation_energies,
            "bandgap_energy_ev": bandgap_energies,
        }
    )

    # Sort by ID to match sample submission structure usually
    df_sub = df_sub.sort_values("id")

    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    df_sub.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    print(df_sub.head())
