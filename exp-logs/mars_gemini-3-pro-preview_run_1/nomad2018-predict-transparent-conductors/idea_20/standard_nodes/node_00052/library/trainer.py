import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from library.config import Config
from library.data_loader import get_dataloaders
from library.model import LAWDS


def train_step(model, dataloader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    total_loss = 0.0
    count = 0

    for atom_x, batch_indices, global_x, targets, _ in dataloader:
        atom_x = atom_x.to(device)
        batch_indices = batch_indices.to(device)
        global_x = global_x.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(atom_x, batch_indices, global_x)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        count += targets.size(0)

    return total_loss / count


def validate_step(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for atom_x, batch_indices, global_x, targets, _ in dataloader:
            atom_x = atom_x.to(device)
            batch_indices = batch_indices.to(device)
            global_x = global_x.to(device)
            targets = targets.to(device)

            outputs = model(atom_x, batch_indices, global_x)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * targets.size(0)
            count += targets.size(0)

    return total_loss / count


def generate_submission(model, test_loader, device, submission_path):
    """
    Generates predictions for the test set and saves them to a CSV file.
    Applies inverse log transformation (exp(x) - 1) to the outputs.
    """
    model.eval()
    ids_list = []
    preds_list = []

    with torch.no_grad():
        for atom_x, batch_indices, global_x, _, ids in test_loader:
            atom_x = atom_x.to(device)
            batch_indices = batch_indices.to(device)
            global_x = global_x.to(device)

            outputs = model(atom_x, batch_indices, global_x)

            # Inverse transform: exp(x) - 1 because targets were log1p transformed
            preds = torch.expm1(outputs).cpu().numpy()

            ids_list.extend(ids.numpy())
            preds_list.append(preds)

    all_preds = np.vstack(preds_list)

    df = pd.DataFrame(
        {
            "id": ids_list,
            "formation_energy_ev_natom": all_preds[:, 0],
            "bandgap_energy_ev": all_preds[:, 1],
        }
    )

    # Sort by ID to ensure consistent order
    df = df.sort_values("id")

    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")


def run_training(load_cached_data=True):
    """
    Main function to run the training pipeline.
    """
    # Set seed for reproducibility
    Config.set_seed(Config.SEED)

    # Setup device
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Load data
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=load_cached_data,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # Initialize model
    model = LAWDS().to(device)

    # Optimization setup
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cite debug_lesson_1: Remove Deprecated `verbose` Parameter from PyTorch Scheduler Initialization
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.SCHEDULER_MIN_LR,
    )

    # Loss function (MSE on log-transformed targets)
    criterion = nn.MSELoss()

    # Training Loop variables
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    print("Starting training...")

    for epoch in range(Config.MAX_EPOCHS):
        train_loss = train_step(model, train_loader, optimizer, criterion, device)
        val_loss = validate_step(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Print metrics (full precision)
        # Note: sqrt(MSE of log targets) approximates RMSLE
        print(
            f"Epoch {epoch+1}/{Config.MAX_EPOCHS} | "
            f"Train Loss (MSE): {train_loss} | "
            f"Val Loss (MSE): {val_loss} | "
            f"Val RMSLE: {np.sqrt(val_loss)} | "
            f"LR: {current_lr}"
        )

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Load best model for inference
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with Val Loss: {best_val_loss}")

    # Generate Submission
    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
