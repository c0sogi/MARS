import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.config import Config
from library.model import ParallelTCNLSTM
from library.data_loader import get_dataloaders


def set_seed(seed=Config.SEED):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_masked_loss(preds, targets, inputs):
    """
    Computes MAE only for the inspiratory phase.
    Handles the fact that u_out is scaled in the inputs.
    """
    # u_out is at index 2 of the features (time_step, u_in, u_out, ...)
    u_out_scaled = inputs[:, :, 2:3]

    # Recover binary mask:
    # u_out raw was 0 (inspiratory) or 1 (expiratory).
    # After scaling, 0 becomes negative (since mean ~0.6).
    # So inspiratory phase corresponds to u_out_scaled < 0.
    mask = (u_out_scaled < 0).float()

    loss = torch.abs(preds - targets) * mask
    # Avoid division by zero
    return loss.sum() / (mask.sum() + 1e-8)


def train_epoch(model, loader, optimizer, device):
    """
    Training logic for one epoch.
    """
    model.train()
    total_loss = 0.0
    steps = 0

    for batch in loader:
        # Unpack batch (X, y)
        X, y = batch
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = model(X)

        # Compute loss
        loss = compute_masked_loss(preds, y, X)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1

    return total_loss / steps if steps > 0 else 0.0


def validate(model, loader, device):
    """
    Validation logic.
    """
    model.eval()
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for batch in loader:
            X, y = batch
            X, y = X.to(device), y.to(device)

            preds = model(X)
            loss = compute_masked_loss(preds, y, X)

            total_loss += loss.item()
            steps += 1

    return total_loss / steps if steps > 0 else 0.0


def generate_submission_file(model, loader, device, save_path):
    """
    Generates predictions for the test set and saves to CSV.
    """
    print("Generating submission...")
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in loader:
            # Test loader returns only X
            X = batch.to(device)
            preds = model(X)
            # Flatten predictions to 1D array
            predictions.append(preds.cpu().numpy().flatten())

    all_preds = np.concatenate(predictions)

    # Load test metadata to get IDs
    # We use the metadata file which is guaranteed to match the order of the test loader
    test_meta = pd.read_csv(Config.TEST_PATH)

    submission = pd.DataFrame({"id": test_meta["id"], "pressure": all_preds})

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    submission.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")


def run_training(config=Config.HYPERPARAMS, save_path=Config.MODEL_SAVE_PATH):
    """
    Main orchestration function for training and evaluation.
    """
    set_seed()
    device = torch.device(config["device"])
    print(f"Using device: {device}")

    # Get DataLoaders
    # This handles caching and preprocessing internally via library.data_loader
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        load_cached_data=not Config.FORCE_RECOMPUTE,
    )

    # Initialize Model
    model = ParallelTCNLSTM(config).to(device)

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["scheduler_factor"],
        patience=config["scheduler_patience"],
        verbose=True,
    )

    # Training Loop
    best_val_mae = float("inf")
    early_stop_counter = 0

    print("Starting training loop...")

    for epoch in range(config["epochs"]):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        # Update learning rate
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{config['epochs']} | Train MAE: {train_loss:.9f} | Val MAE: {val_loss:.9f}"
        )

        # Checkpointing and Early Stopping
        if val_loss < best_val_mae:
            best_val_mae = val_loss
            torch.save(model.state_dict(), save_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= config["patience"]:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val MAE: {best_val_mae:.9f}")

    # Load best model for submission
    model.load_state_dict(torch.load(save_path))

    # Generate Submission
    generate_submission_file(model, test_loader, device, Config.SUBMISSION_PATH)
