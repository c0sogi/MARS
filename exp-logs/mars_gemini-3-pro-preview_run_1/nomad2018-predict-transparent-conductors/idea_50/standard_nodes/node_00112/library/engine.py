import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.model import AMSA_DS
from library.data import get_train_val_loaders, get_test_loader
from library.utils import set_seed, log1p_transform, expm1_transform


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (atomic_feats, global_feats, batch_indices, targets, _) in enumerate(
        loader
    ):
        # Move data to device
        atomic_feats = atomic_feats.to(device)
        global_feats = global_feats.to(device)
        batch_indices = batch_indices.to(device)
        targets = targets.to(device)

        # Transform targets to log scale
        log_targets = torch.log1p(targets)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(atomic_feats, global_feats, batch_indices)

        # Compute loss
        loss = criterion(outputs, log_targets)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        running_loss += loss.item() * targets.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for atomic_feats, global_feats, batch_indices, targets, _ in loader:
            atomic_feats = atomic_feats.to(device)
            global_feats = global_feats.to(device)
            batch_indices = batch_indices.to(device)
            targets = targets.to(device)

            # Transform targets to log scale for consistent loss calculation
            log_targets = torch.log1p(targets)

            outputs = model(atomic_feats, global_feats, batch_indices)
            loss = criterion(outputs, log_targets)

            running_loss += loss.item() * targets.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def run_training(load_cached_data=True):
    """
    Main training loop with Early Stopping and Scheduler.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Data Loaders
    train_loader, val_loader = get_train_val_loaders(load_cached_data=load_cached_data)

    # Model
    model = AMSA_DS().to(device)

    # Optimizer & Criterion
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.MSELoss()

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.SCHEDULER_MIN_LR,
    )

    # Early Stopping variables
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.EXECUTION_DIR, "best_model.pt")

    print(f"Starting training on {device}...")
    print(
        f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}"
    )

    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"LR: {current_lr:.2e}"
        )

        # Check Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            # print(f"  -> New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"Training complete. Best Val Loss: {best_val_loss:.6f}")
    return best_val_loss


def generate_submission(load_cached_data=True):
    """
    Generates predictions for the test set using the best trained model.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Load Test Data
    test_loader = get_test_loader(load_cached_data=load_cached_data)

    # Load Model
    model = AMSA_DS().to(device)
    model_path = os.path.join(Config.EXECUTION_DIR, "best_model.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Best model not found at {model_path}. Run training first."
        )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_ids = []
    all_preds = []

    print("Generating predictions...")
    with torch.no_grad():
        for atomic_feats, global_feats, batch_indices, _, batch_ids in test_loader:
            atomic_feats = atomic_feats.to(device)
            global_feats = global_feats.to(device)
            batch_indices = batch_indices.to(device)

            # Forward pass (predicts log-scale values)
            outputs = model(atomic_feats, global_feats, batch_indices)

            # Inverse transform to original scale
            preds_original = torch.expm1(outputs)

            all_ids.extend(batch_ids.cpu().numpy())
            all_preds.extend(preds_original.cpu().numpy())

    # Create DataFrame
    submission_df = pd.DataFrame(
        all_preds, columns=["formation_energy_ev_natom", "bandgap_energy_ev"]
    )
    submission_df.insert(0, "id", all_ids)

    # Save submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
