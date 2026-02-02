import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders
from library.model import CrystalGraphResNet


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to run on.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch)

        # Compute loss
        loss = criterion(outputs, batch.y)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch.num_graphs

    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device, scaler=None):
    """
    Evaluates the model on a validation/test set.

    Args:
        model: The PyTorch model.
        loader: DataLoader for evaluation data.
        criterion: Loss function.
        device: Device to run on.
        scaler: Optional TargetScaler to compute metrics on original scale.

    Returns:
        dict: Dictionary containing loss and other metrics.
    """
    model.eval()
    running_loss = 0.0

    # Lists to store predictions and targets for unscaled metrics
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)
            loss = criterion(outputs, batch.y)
            running_loss += loss.item() * batch.num_graphs

            if scaler is not None:
                all_preds.append(outputs.cpu())
                all_targets.append(batch.y.cpu())

    avg_loss = running_loss / len(loader.dataset)
    metrics = {"loss": avg_loss}

    if scaler is not None and len(all_preds) > 0:
        # Concatenate and inverse transform
        preds_scaled = torch.cat(all_preds, dim=0)
        targets_scaled = torch.cat(all_targets, dim=0)

        preds_orig = scaler.inverse_transform(preds_scaled)
        targets_orig = scaler.inverse_transform(targets_scaled)

        # Compute MAE on original scale for each target
        mae = torch.mean(torch.abs(preds_orig - targets_orig), dim=0)
        metrics["mae_formation"] = mae[0].item()
        metrics["mae_bandgap"] = mae[1].item()

        # Compute MSE on original scale
        mse = torch.mean((preds_orig - targets_orig) ** 2, dim=0)
        metrics["mse_formation"] = mse[0].item()
        metrics["mse_bandgap"] = mse[1].item()

    return metrics


def predict(model, loader, device, scaler):
    """
    Generates predictions for the dataset.

    Args:
        model: The PyTorch model.
        loader: DataLoader.
        device: Device.
        scaler: TargetScaler to inverse transform predictions.

    Returns:
        tuple: (ids, predictions) where predictions are in original scale.
    """
    model.eval()
    ids_list = []
    preds_list = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)

            # Inverse transform to get original units (eV)
            outputs_orig = scaler.inverse_transform(outputs)

            ids_list.append(batch.id.cpu())
            preds_list.append(outputs_orig.cpu())

    return (
        torch.cat(ids_list, dim=0).numpy().flatten(),
        torch.cat(preds_list, dim=0).numpy(),
    )


def run_training(
    num_epochs=Config.NUM_EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    load_cached_data=True,
):
    """
    Main function to run the training pipeline.

    Args:
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        learning_rate (float): Learning rate.
        weight_decay (float): Weight decay for optimizer.
        load_cached_data (bool): Whether to load pre-processed graphs from cache.
    """
    # Set reproducibility
    set_seed(Config.SEED)

    # Prepare directories
    Config.prepare_directories()

    # Get DataLoaders and Scaler
    train_loader, val_loader, test_loader, scaler = get_dataloaders(
        batch_size=batch_size,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=load_cached_data,
    )

    # Initialize Model
    device = torch.device(Config.DEVICE)
    model = CrystalGraphResNet(config=Config).to(device)

    # Optimizer and Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.MSELoss()

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, verbose=True
    )

    # Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training on {device} for {num_epochs} epochs...")

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device, scaler)
        val_loss = val_metrics["loss"]

        # Update Scheduler
        scheduler.step(val_loss)

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            saved_str = " [Saved Best]"
        else:
            patience_counter += 1
            saved_str = ""

        epoch_time = time.time() - start_time

        print(
            f"Epoch {epoch:03d} | Time: {epoch_time:.2f}s | "
            f"Train Loss: {train_loss:.8f} | Val Loss: {val_loss:.8f} | "
            f"Val MAE Form: {val_metrics.get('mae_formation', 0):.6f} | "
            f"Val MAE Gap: {val_metrics.get('mae_bandgap', 0):.6f}{saved_str}"
        )

        # Early Stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    # Load best model for inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Generate Predictions
    print("Generating predictions on test set...")
    ids, preds = predict(model, test_loader, device, scaler)

    # Create Submission DataFrame
    submission_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": preds[:, 0],
            "bandgap_energy_ev": preds[:, 1],
        }
    )

    # Sort by ID to ensure correct order
    submission_df = submission_df.sort_values("id")

    # Save Submission
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    return submission_df
