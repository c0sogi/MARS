import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.model import LI_CGCNN_ELR
from library.data import get_dataloaders
from library.utils import set_seed, save_checkpoint, load_checkpoint, StandardScaler


def train_one_epoch(model, loader, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        preds = model(batch)

        # Compute loss (MSE on standardized targets)
        loss = nn.MSELoss()(preds, batch.y)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch.num_graphs

    return running_loss / len(loader.dataset)


def evaluate(model, loader, device):
    """
    Evaluates the model on a validation or test set.
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # Forward pass
            preds = model(batch)

            # Compute loss (MSE on standardized targets)
            # Note: For test set, batch.y contains NaNs, so loss would be NaN.
            # This function is primarily for validation where y is known.
            if not torch.isnan(batch.y).any():
                loss = nn.MSELoss()(preds, batch.y)
                running_loss += loss.item() * batch.num_graphs

    return running_loss / len(loader.dataset)


def generate_submission(model, test_loader, device):
    """
    Generates predictions for the test set and saves to submission.csv.
    """
    print("Generating submission...")
    model.eval()
    all_preds = []

    # Load target scaler to inverse transform predictions
    target_scaler = StandardScaler()
    if os.path.exists(Config.TARGET_SCALER_PATH):
        target_scaler.load(Config.TARGET_SCALER_PATH)
    else:
        raise FileNotFoundError(
            f"Target scaler not found at {Config.TARGET_SCALER_PATH}. Train model first."
        )

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            preds_scaled = model(batch)

            # Move to CPU and inverse transform
            preds_scaled_np = preds_scaled.cpu().numpy()
            preds_original = target_scaler.inverse_transform(preds_scaled_np)

            all_preds.append(preds_original)

    # Concatenate all predictions
    all_preds = np.concatenate(all_preds, axis=0)

    # Load test metadata to get IDs
    test_meta = pd.read_csv(Config.TEST_METADATA_PATH)
    ids = test_meta["id"].values

    # Ensure lengths match
    if len(ids) != len(all_preds):
        raise ValueError(
            f"Mismatch between number of IDs ({len(ids)}) and predictions ({len(all_preds)})"
        )

    # Create submission DataFrame
    submission_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": all_preds[:, 0],
            "bandgap_energy_ev": all_preds[:, 1],
        }
    )

    # Save to CSV
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_training(load_cached_data=True, num_epochs=None):
    """
    Main function to run the training pipeline.
    """
    # Set reproducibility
    set_seed(Config.SEED)

    # Setup device
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Get DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # Initialize Model
    model = LI_CGCNN_ELR(Config).to(device)

    # Initialize Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Training Loop configuration
    epochs = num_epochs if num_epochs is not None else Config.NUM_EPOCHS
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validate
        val_loss = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.8f} - Val Loss: {val_loss:.8f}"
        )

        # Early Stopping and Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_loss, Config.BEST_MODEL_PATH)
            # print(f"  New best model saved! (Val Loss: {val_loss:.8f})")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    # Load best model for submission
    print(f"Loading best model from {Config.BEST_MODEL_PATH}")
    epoch, loss = load_checkpoint(
        model, None, Config.BEST_MODEL_PATH, device=Config.DEVICE
    )
    print(f"Loaded model from epoch {epoch} with val loss {loss:.8f}")

    # Generate Submission
    generate_submission(model, test_loader, device)

    return model
