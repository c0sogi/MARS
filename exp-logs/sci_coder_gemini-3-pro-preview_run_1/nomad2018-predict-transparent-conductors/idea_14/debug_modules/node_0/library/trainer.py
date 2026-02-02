import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import time
from library.config import Config, set_seed
from library.data_loader import get_data_loaders
from library.model import DCPDS_Model


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch in loader:
        # Unpack batch
        atomic_feats, batch_idx, global_feats, targets, _ = batch

        # Move to device
        atomic_feats = atomic_feats.to(device)
        batch_idx = batch_idx.to(device)
        global_feats = global_feats.to(device)
        targets = targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(atomic_feats, batch_idx, global_feats)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss (weighted by batch size for accurate mean)
        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / total_samples


def validate_epoch(model, loader, criterion, device):
    """
    Performs validation on the given loader.
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            atomic_feats, batch_idx, global_feats, targets, _ = batch

            atomic_feats = atomic_feats.to(device)
            batch_idx = batch_idx.to(device)
            global_feats = global_feats.to(device)
            targets = targets.to(device)

            outputs = model(atomic_feats, batch_idx, global_feats)
            loss = criterion(outputs, targets)

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

    return running_loss / total_samples


def generate_predictions(model, loader, device, output_path):
    """
    Generates predictions for the test set and saves them to a CSV file.
    """
    model.eval()
    results = []

    print(f"Generating predictions for {len(loader.dataset)} samples...")

    with torch.no_grad():
        for batch in loader:
            atomic_feats, batch_idx, global_feats, _, ids = batch

            atomic_feats = atomic_feats.to(device)
            batch_idx = batch_idx.to(device)
            global_feats = global_feats.to(device)

            outputs = model(atomic_feats, batch_idx, global_feats)

            # Inverse transform: exp(x) - 1 (reversing log1p)
            preds = torch.expm1(outputs).cpu().numpy()

            for i, sample_id in enumerate(ids):
                results.append(
                    {
                        "id": sample_id,
                        "formation_energy_ev_natom": preds[i, 0],
                        "bandgap_energy_ev": preds[i, 1],
                    }
                )

    # Create DataFrame and save
    df = pd.DataFrame(results)
    # Ensure correct column order
    df = df[["id", "formation_energy_ev_natom", "bandgap_energy_ev"]]

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(config=Config, debug_sample_size=None):
    """
    Main function to run the training pipeline.
    """
    # Set random seed
    set_seed(config.SEED)
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # Load Data
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=config.BATCH_SIZE,
        load_cached_data=True,
        debug_sample_size=debug_sample_size,
    )

    # Initialize Model
    print("Initializing Model...")
    model = DCPDS_Model(config).to(device)

    # Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE,
        min_lr=config.SCHEDULER_MIN_LR,
        verbose=True,
    )

    # Loss Function (MSE on log-transformed targets)
    criterion = nn.MSELoss()

    # Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training for {config.NUM_EPOCHS} epochs...")
    start_time = time.time()

    for epoch in range(config.NUM_EPOCHS):
        epoch_start = time.time()

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)

        # Scheduler step
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train MSE: {train_loss:.8f} | "
            f"Val MSE: {val_loss:.8f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_time:.2f}s"
        )

        # Early Stopping and Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            # print(f"  -> New best model saved (Val MSE: {val_loss:.8f})")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    total_time = time.time() - start_time
    print(
        f"Training finished in {total_time:.2f}s. Best Validation MSE: {best_val_loss:.8f}"
    )

    # Load best model for inference
    print("Loading best model for inference...")
    if os.path.exists(config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Warning: No model checkpoint found. Using current model state.")

    # Generate Submission
    generate_predictions(model, test_loader, device, config.SUBMISSION_PATH)
