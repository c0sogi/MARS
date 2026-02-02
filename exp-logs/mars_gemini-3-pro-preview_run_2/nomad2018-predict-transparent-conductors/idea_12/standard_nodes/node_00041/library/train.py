import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import (
    CHECKPOINT_DIR,
    SUBMISSION_DIR,
    DEVICE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    PATIENCE,
    TARGET_COLS,
    SEED,
)
from library.model import CrystalGraphConvNet
from library.data import get_dataloaders
from library.utils import rmsle


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model, loader, optimizer, criterion, device, comp_scaler, target_scaler
):
    model.train()
    running_loss = 0.0

    for batch in loader:
        batch = batch.to(device)

        # Prepare inputs
        # Composition is no longer used in the model

        # Scale targets for training stability (Log1p + Standardization)
        targets = target_scaler.transform(batch.y)

        optimizer.zero_grad()

        # Forward pass
        preds = model(batch)

        # Compute loss in the scaled space
        loss = criterion(preds, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch.num_graphs

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device, comp_scaler, target_scaler):
    model.eval()
    running_loss = 0.0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # Targets for loss calculation
            targets_scaled = target_scaler.transform(batch.y)

            # Forward pass
            preds_scaled = model(batch)

            # Loss in scaled space
            loss = criterion(preds_scaled, targets_scaled)
            running_loss += loss.item() * batch.num_graphs

            # Inverse transform for metric calculation (back to eV)
            preds_original = target_scaler.inverse_transform(preds_scaled)
            targets_original = batch.y  # Original values are in batch.y

            all_preds.append(preds_original.cpu().numpy())
            all_targets.append(targets_original.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    # Concatenate all predictions and targets
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate RMSLE on original scale
    # Ensure non-negative for log
    all_preds = np.maximum(all_preds, 0)
    all_targets = np.maximum(all_targets, 0)

    val_rmsle = rmsle(all_targets, all_preds)

    return epoch_loss, val_rmsle


def generate_submission(model, loader, device, comp_scaler, target_scaler, output_path):
    model.eval()
    ids = []
    preds_list = []

    print("Generating submission...")
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # Forward pass
            preds_scaled = model(batch)

            # Inverse transform to get original units
            preds_original = target_scaler.inverse_transform(preds_scaled)

            # Collect IDs and predictions
            # batch.id is a list or tensor of IDs
            if isinstance(batch.id, torch.Tensor):
                ids.extend(batch.id.cpu().numpy().tolist())
            else:
                ids.extend(batch.id)

            preds_list.append(preds_original.cpu().numpy())

    # Concatenate predictions
    all_preds = np.concatenate(preds_list, axis=0)

    # Create DataFrame
    submission_df = pd.DataFrame(
        {"id": ids, TARGET_COLS[0]: all_preds[:, 0], TARGET_COLS[1]: all_preds[:, 1]}
    )

    # Sort by ID to match sample submission format
    submission_df = submission_df.sort_values("id")

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    patience=PATIENCE,
    load_cached_data=True,
):
    set_seed(SEED)

    # Load data
    train_loader, val_loader, test_loader, comp_scaler, target_scaler = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # Initialize model
    model = CrystalGraphConvNet().to(DEVICE)

    # Optimizer and Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.MSELoss()

    # Training Loop
    best_val_rmsle = float("inf")
    epochs_no_improve = 0
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training on {DEVICE} for {num_epochs} epochs...")

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            DEVICE,
            comp_scaler,
            target_scaler,
        )

        # Validate
        val_loss, val_rmsle = validate(
            model, val_loader, criterion, DEVICE, comp_scaler, target_scaler
        )

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val RMSLE: {val_rmsle:.10f} | "
            f"Time: {elapsed:.2f}s"
        )

        # Early Stopping and Checkpointing
        if val_rmsle < best_val_rmsle:
            best_val_rmsle = val_rmsle
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved! RMSLE: {best_val_rmsle:.10f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"Early stopping triggered after {patience} epochs without improvement."
                )
                break

    print(f"Training complete. Best Validation RMSLE: {best_val_rmsle:.10f}")

    # Load best model for submission
    model.load_state_dict(torch.load(best_model_path))

    # Generate Submission
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    generate_submission(
        model, test_loader, DEVICE, comp_scaler, target_scaler, submission_path
    )
