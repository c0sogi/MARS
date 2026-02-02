import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import set_seed, rmsle
from library.data import get_loaders
from library.model import CGCNN_IB


def train_one_epoch(model, loader, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss()

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        preds = model(batch)

        # Compute loss (targets are already standardized in the loader)
        loss = criterion(preds, batch.y)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


def validate(model, loader, device, scaler):
    """
    Evaluates the model on the validation set using the competition metric (RMSLE).
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds = model(batch)

            # Inverse transform to get original units (eV)
            # Both preds and batch.y are (Batch, 2) tensors
            preds_original = scaler.inverse_transform(preds)
            targets_original = scaler.inverse_transform(batch.y)

            all_preds.append(preds_original.cpu())
            all_targets.append(targets_original.cpu())

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate RMSLE
    # Note: rmsle utility expects tensors or numpy arrays
    score = rmsle(all_targets, all_preds)
    return score


def generate_submission(model, loader, device, scaler, output_path):
    """
    Generates predictions for the test set and saves to CSV.
    """
    model.eval()
    ids = []
    preds_formation = []
    preds_bandgap = []

    print(f"Generating submission at {output_path}...")

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds = model(batch)

            # Inverse transform predictions
            preds_original = scaler.inverse_transform(preds)

            # Collect IDs and predictions
            # batch.id is a list or tensor of IDs
            ids.extend(batch.id.cpu().numpy().flatten())
            preds_formation.extend(preds_original[:, 0].cpu().numpy().flatten())
            preds_bandgap.extend(preds_original[:, 1].cpu().numpy().flatten())

    # Create DataFrame
    df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": preds_formation,
            "bandgap_energy_ev": preds_bandgap,
        }
    )

    # Sort by ID to ensure correct order (though not strictly required if IDs match)
    df = df.sort_values("id")

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print("Submission saved.")


def run_training(load_cached_data=True):
    """
    Main function to run the training pipeline.
    """
    set_seed(Config.RANDOM_SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader, scaler = get_loaders(
        load_cached_data=load_cached_data
    )

    # 2. Model Initialization
    print("Initializing Model...")
    model = CGCNN_IB(config=Config).to(device)

    # 3. Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Reduce LR if validation metric stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, verbose=False
    )

    # 4. Training Loop
    best_val_score = float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validate
        val_score = validate(model, val_loader, device, scaler)

        # Scheduler Step
        scheduler.step(val_score)

        epoch_time = time.time() - start_time

        print(
            f"Epoch {epoch:03d} | Time: {epoch_time:.2f}s | "
            f"Train Loss (MSE): {train_loss:.6f} | "
            f"Val RMSLE: {val_score:.10f}"
        )

        # Early Stopping and Checkpointing
        if val_score < best_val_score:
            best_val_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            # print(f"  New best model saved! RMSLE: {best_val_score:.10f}")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    print(f"Training complete. Best Validation RMSLE: {best_val_score:.10f}")

    # 5. Generate Submission
    # Load best model
    print("Loading best model for submission...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    generate_submission(model, test_loader, device, scaler, Config.SUBMISSION_PATH)
