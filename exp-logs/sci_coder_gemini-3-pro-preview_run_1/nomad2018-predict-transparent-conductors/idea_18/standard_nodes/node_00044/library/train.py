import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd

from library.config import Config
from library.data import CrystalDataset, collate_crystals
from library.model import CRNDSModel


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Move batch to device
        atomic_features = batch["atomic_features"].to(device)
        global_features = batch["global_features"].to(device)
        batch_index = batch["batch_index"].to(device)
        targets = batch["targets"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(atomic_features, global_features, batch_index)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
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
        for batch in loader:
            atomic_features = batch["atomic_features"].to(device)
            global_features = batch["global_features"].to(device)
            batch_index = batch["batch_index"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(atomic_features, global_features, batch_index)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * targets.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def run_training(
    load_cached_data=True,
    num_epochs=Config.NUM_EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    patience=Config.EARLY_STOPPING_PATIENCE,
):
    """
    Orchestrates the training process.
    """
    # Ensure working directory exists
    Config.setup()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print("Initializing datasets...")
    # Initialize Train Dataset (fits and saves scalers)
    train_dataset = CrystalDataset(
        metadata_path=Config.TRAIN_META_PATH,
        cache_path=Config.TRAIN_CACHE_PATH,
        scalers_path=Config.SCALERS_CACHE_PATH,
        split="train",
        load_cached_data=load_cached_data,
    )

    # Initialize Validation Dataset (loads scalers)
    val_dataset = CrystalDataset(
        metadata_path=Config.VAL_META_PATH,
        cache_path=Config.VAL_CACHE_PATH,
        scalers_path=Config.SCALERS_CACHE_PATH,
        split="val",
        load_cached_data=load_cached_data,
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_crystals,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_crystals,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Model, Criterion, Optimizer
    print("Initializing model...")
    model = CRNDSModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # Training Loop with Early Stopping
    best_val_loss = float("inf")
    epochs_no_improve = 0

    print("Starting training...")
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        # Scheduler step
        scheduler.step(val_loss)

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
            print(
                f"Epoch {epoch+1}: New best model saved with Val Loss: {val_loss:.8f}"
            )
        else:
            epochs_no_improve += 1

        epoch_time = time.time() - epoch_start

        # Calculate RMSLE (since we trained on logs, sqrt(MSE) is RMSLE)
        train_rmsle = np.sqrt(train_loss)
        val_rmsle = np.sqrt(val_loss)

        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss (MSE): {train_loss:.8f} (RMSLE: {train_rmsle:.8f}) - "
            f"Val Loss (MSE): {val_loss:.8f} (RMSLE: {val_rmsle:.8f}) - "
            f"Time: {epoch_time:.2f}s"
        )

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    total_time = time.time() - start_time
    print(
        f"Training completed in {total_time:.2f}s. Best Val Loss: {best_val_loss:.8f}"
    )


def generate_submission(load_cached_data=True, batch_size=Config.BATCH_SIZE):
    """
    Loads the best model, predicts on the test set, and saves the submission file.
    """
    Config.setup()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print("Initializing test dataset...")
    test_dataset = CrystalDataset(
        metadata_path=Config.TEST_META_PATH,
        cache_path=Config.TEST_CACHE_PATH,
        scalers_path=Config.SCALERS_CACHE_PATH,
        split="test",
        load_cached_data=load_cached_data,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_crystals,
        num_workers=2,
    )

    print("Loading best model...")
    model = CRNDSModel().to(device)
    if os.path.exists(Config.MODEL_CHECKPOINT_PATH):
        model.load_state_dict(
            torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=device)
        )
    else:
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.MODEL_CHECKPOINT_PATH}"
        )

    model.eval()

    ids = []
    predictions = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            atomic_features = batch["atomic_features"].to(device)
            global_features = batch["global_features"].to(device)
            batch_index = batch["batch_index"].to(device)
            batch_ids = batch["ids"]

            # Forward pass
            outputs = model(atomic_features, global_features, batch_index)

            # Inverse transform: exp(x) - 1
            # Since targets were log(1+y)
            preds_original_scale = torch.expm1(outputs)

            ids.extend(batch_ids)
            predictions.append(preds_original_scale.cpu().numpy())

    # Concatenate all predictions
    predictions = np.vstack(predictions)

    # Create DataFrame
    submission_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": predictions[:, 0],
            "bandgap_energy_ev": predictions[:, 1],
        }
    )

    # Sort by ID to ensure correct order
    submission_df.sort_values("id", inplace=True)

    # Save submission
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission_df.head())
