import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time

from library.config import Config
from library.utils import set_seed, save_checkpoint, StandardScaler
from library.data import get_dataloaders
from library.model import RA_CGN_AR


def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    num_samples = 0

    for batch in loader:
        batch = batch.to(device)

        # Standardize targets
        targets_normalized = scaler.transform(batch.y)

        # Forward pass
        outputs = model(batch)

        # Compute loss
        loss = criterion(outputs, targets_normalized)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch.num_graphs
        num_samples += batch.num_graphs

    return running_loss / num_samples


def evaluate(model, loader, criterion, scaler, device):
    """
    Evaluates the model on a given dataset (validation).
    Returns the average loss (MSE on standardized targets) and
    MAE (on original scale).
    """
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # Standardize targets for loss calculation
            targets_normalized = scaler.transform(batch.y)

            # Forward pass
            outputs = model(batch)

            # Loss (MSE on standardized data)
            loss = criterion(outputs, targets_normalized)

            # Inverse transform for physical metrics
            outputs_original = scaler.inverse_transform(outputs)
            targets_original = batch.y

            # MAE on original scale
            mae = torch.abs(outputs_original - targets_original).mean()

            running_loss += loss.item() * batch.num_graphs
            running_mae += mae.item() * batch.num_graphs
            num_samples += batch.num_graphs

    avg_loss = running_loss / num_samples
    avg_mae = running_mae / num_samples

    return avg_loss, avg_mae


def run_training(subset_size=None, load_cached_data=True):
    """
    Main function to run the training pipeline.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Running training on device: {device}")

    # 1. Prepare Data
    train_loader, val_loader, _ = get_dataloaders(
        subset_size=subset_size, load_cached_data=load_cached_data
    )

    # 2. Prepare Scaler
    print("Fitting StandardScaler on training data...")
    # Collect all training targets to fit the scaler
    all_train_targets = []
    for data in train_loader.dataset:
        all_train_targets.append(data.y)
    all_train_targets = torch.cat(all_train_targets, dim=0)

    scaler = StandardScaler(device=device)
    scaler.fit(all_train_targets)

    # Save scaler for inference
    scaler_path = os.path.join(Config.CACHE_DIR, "target_scaler.npz")
    scaler.save(scaler_path)
    print(f"Scaler saved to {scaler_path}")

    # 3. Initialize Model, Optimizer, Loss
    model = RA_CGN_AR().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.MSELoss()

    # Scheduler: Reduce LR when validation loss stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # 4. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")
    print(
        f"{'Epoch':<6} | {'Train Loss':<12} | {'Val Loss':<12} | {'Val MAE':<12} | {'Time':<8}"
    )
    print("-" * 60)

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device
        )

        # Validate
        val_loss, val_mae = evaluate(model, val_loader, criterion, scaler, device)

        # Update Scheduler
        scheduler.step(val_loss)

        elapsed = time.time() - start_time

        print(
            f"{epoch:<6} | {train_loss:.8f}   | {val_loss:.8f}   | {val_mae:.8f}   | {elapsed:.1f}s"
        )

        # Checkpoint & Early Stopping
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_loss": best_val_loss,
                },
                is_best=True,
                checkpoint_dir=Config.CHECKPOINT_DIR,
            )
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break

    print(f"\nTraining complete. Best Validation Loss: {best_val_loss:.8f}")
