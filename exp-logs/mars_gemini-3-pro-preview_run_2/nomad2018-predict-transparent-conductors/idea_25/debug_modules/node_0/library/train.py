import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from library.config import (
    CHECKPOINT_DIR,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    SCHEDULER_MIN_LR,
    CACHE_DIR,
    RANDOM_SEED,
)
from library.utils import set_seed, compute_rmsle, get_scaler
from library.data import get_train_val_datasets
from library.model import MH_RA_CGN


def train_epoch(model, loader, optimizer, criterion, scaler, device):
    """
    Performs one epoch of training.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        # Forward pass
        out = model(data)

        # Scale targets for loss calculation
        # data.y is (Batch, 2)
        targets_scaled = scaler.transform(data.y)

        # Compute loss
        loss = criterion(out, targets_scaled)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, loader, criterion, scaler, device):
    """
    Evaluates the model on the validation set.
    Returns:
        avg_mse_loss: MSE loss on scaled targets (for scheduler).
        avg_rmsle: RMSLE on original scale targets (for model selection).
    """
    model.eval()
    total_mse_loss = 0.0
    all_preds_raw = []
    all_targets_raw = []
    num_batches = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            # Forward pass
            out = model(data)

            # 1. Compute MSE on scaled targets (consistent with training loss)
            targets_scaled = scaler.transform(data.y)
            loss = criterion(out, targets_scaled)
            total_mse_loss += loss.item()

            # 2. Prepare for RMSLE on raw scale
            # Inverse transform predictions to eV
            preds_raw = scaler.inverse_transform(out)

            all_preds_raw.append(preds_raw)
            all_targets_raw.append(data.y)

            num_batches += 1

    avg_mse_loss = total_mse_loss / num_batches if num_batches > 0 else 0.0

    # Concatenate all batches
    if len(all_preds_raw) > 0:
        all_preds_raw = torch.cat(all_preds_raw, dim=0)
        all_targets_raw = torch.cat(all_targets_raw, dim=0)

        # Compute RMSLE
        avg_rmsle = compute_rmsle(all_preds_raw, all_targets_raw)
    else:
        avg_rmsle = 0.0

    return avg_mse_loss, avg_rmsle


def train_model(load_cached_data=True):
    """
    Main training loop.
    """
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading datasets...")
    train_dataset, val_dataset = get_train_val_datasets(load_cached=load_cached_data)

    # 2. Prepare Scaler
    print("Fitting/Loading target scaler...")
    # Collect all training targets to fit scaler
    # Note: This might be memory intensive for huge datasets, but fine here.
    # Alternatively, could compute mean/std iteratively.
    all_train_targets = torch.cat([d.y for d in train_dataset], dim=0)
    scaler_path = os.path.join(CACHE_DIR, "target_scaler.npz")
    scaler = get_scaler(
        all_train_targets, scaler_path, load_cached_data=load_cached_data
    )

    # Move scaler stats to device for efficient transform during loop
    if isinstance(scaler.mean, torch.Tensor):
        scaler.mean = scaler.mean.to(device)
        scaler.std = scaler.std.to(device)
    else:
        scaler.mean = torch.tensor(scaler.mean, device=device, dtype=torch.float32)
        scaler.std = torch.tensor(scaler.std, device=device, dtype=torch.float32)

    # 3. Create Loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Initialize Model
    model = MH_RA_CGN().to(device)

    # 5. Optimizer & Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=SCHEDULER_MIN_LR,
        verbose=True,
    )
    criterion = nn.MSELoss()

    # 6. Training Loop
    best_val_rmsle = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training for {NUM_EPOCHS} epochs...")
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device
        )

        # Validate
        val_mse, val_rmsle = validate(model, val_loader, criterion, scaler, device)

        # Scheduler step (using MSE loss as it's the direct optimization objective)
        scheduler.step(val_mse)

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch:03d} | "
            f"Train MSE: {train_loss:.6f} | "
            f"Val MSE: {val_mse:.6f} | "
            f"Val RMSLE: {val_rmsle:.6f} | "
            f"Time: {epoch_time:.2f}s"
        )

        # Early Stopping check (using RMSLE as it's the target metric)
        if val_rmsle < best_val_rmsle:
            best_val_rmsle = val_rmsle
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            # print(f"  New best model saved! RMSLE: {best_val_rmsle:.6f}")
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    total_time = time.time() - start_time
    print(f"Training complete. Total time: {total_time:.2f}s")
    print(f"Best Validation RMSLE: {best_val_rmsle:.6f}")

    return model
