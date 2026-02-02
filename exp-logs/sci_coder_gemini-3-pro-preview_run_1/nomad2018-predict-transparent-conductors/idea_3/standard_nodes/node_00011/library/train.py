import os
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.data import get_datasets, collate_fn
from library.model import RBFDualStreamDeepSets


def set_seed(seed=42):
    """Sets random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model: PyTorch model
        dataloader: Training DataLoader
        criterion: Loss function
        optimizer: Optimizer
        device: Calculation device

    Returns:
        Average training loss for the epoch
    """
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        # Move batch data to device
        atomic_features = batch["atomic_features"].to(device)
        lattice_features = batch["lattice_features"].to(device)
        batch_indices = batch["batch_indices"].to(device)
        targets = batch["targets"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(atomic_features, lattice_features, batch_indices)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Accumulate loss (MSELoss is mean, so multiply by batch size)
        running_loss += loss.item() * targets.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Validates the model on the validation set.

    Args:
        model: PyTorch model
        dataloader: Validation DataLoader
        criterion: Loss function
        device: Calculation device

    Returns:
        tuple: (Average MSE Loss, Average Column-wise RMSLE)
    """
    model.eval()
    running_loss = 0.0

    # For column-wise metrics
    squared_errors = torch.zeros(2).to(device)
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            atomic_features = batch["atomic_features"].to(device)
            lattice_features = batch["lattice_features"].to(device)
            batch_indices = batch["batch_indices"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(atomic_features, lattice_features, batch_indices)

            loss = criterion(outputs, targets)
            running_loss += loss.item() * targets.size(0)

            # Accumulate squared errors per column for detailed metric
            # Since targets are already log(1+y), MSE on them is MSLE
            errors = (outputs - targets) ** 2
            squared_errors += errors.sum(dim=0)
            total_samples += targets.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)

    # Calculate column-wise RMSLE
    # RMSLE = sqrt(mean((log(1+p) - log(1+t))^2))
    # Since our model predicts log(1+p) directly against log(1+t),
    # this is just sqrt(MSE) per column.
    mse_per_col = squared_errors / total_samples
    rmsle_per_col = torch.sqrt(mse_per_col)
    avg_rmsle = torch.mean(rmsle_per_col).item()

    return epoch_loss, avg_rmsle


def train_model(
    num_epochs=Config.NUM_EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    patience=Config.PATIENCE,
    load_cached_data=True,
):
    """
    Main training loop with early stopping and model checkpointing.
    """
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    # The get_datasets function handles calling process_dataset which handles caching
    train_dataset, val_dataset, _ = get_datasets(load_cached_data=load_cached_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # 2. Initialize Model
    model = RBFDualStreamDeepSets().to(device)

    # 3. Setup Training Components
    # Using MSE Loss on log-transformed targets
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # 4. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_rmsle = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {train_loss} - "
            f"Val Loss: {val_loss} - "
            f"Val RMSLE: {val_rmsle} - "
            f"Time: {time.time() - start_time:.2f}s"
        )

        # Early Stopping and Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"New best model saved with Val Loss: {val_loss}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    return best_val_loss


def generate_submission(
    model_path=Config.MODEL_SAVE_PATH,
    batch_size=Config.BATCH_SIZE,
    load_cached_data=True,
):
    """
    Generates predictions for the test set and saves to submission.csv.
    """
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data
    # We need the scaler from the training set (handled internally by get_datasets),
    # so we call get_datasets but only use the test_dataset.
    _, _, test_dataset = get_datasets(load_cached_data=load_cached_data)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    # 2. Load Model
    model = RBFDualStreamDeepSets().to(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Inference
    ids = []
    predictions = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            atomic_features = batch["atomic_features"].to(device)
            lattice_features = batch["lattice_features"].to(device)
            batch_indices = batch["batch_indices"].to(device)

            # Forward pass
            outputs = model(atomic_features, lattice_features, batch_indices)

            # Inverse transform: y = exp(y') - 1
            # Ensure non-negative results just in case
            preds = torch.expm1(outputs)
            preds = torch.clamp(preds, min=0.0)

            ids.extend(batch["ids"].numpy())
            predictions.extend(preds.cpu().numpy())

    # 4. Save Submission
    predictions = np.array(predictions)
    submission_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": predictions[:, 0],
            "bandgap_energy_ev": predictions[:, 1],
        }
    )

    # Ensure correct column order
    submission_df = submission_df[
        ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
    ]

    # Save
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    # Preview
    print(submission_df.head())
