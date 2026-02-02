import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import (
    SEED,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    TRAIN_CACHE_PATH,
    VAL_CACHE_PATH,
    TEST_CACHE_PATH,
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    DEVICE,
    NUM_WORKERS,
    setup_directories,
)
from library.data_utils import (
    process_data,
    CrystalDataset,
    collate_fn,
    StandardScaler,
)
from library.model import CADSTFModel


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Move data to device
        global_feats = batch["global_features"].to(device)
        atomic_feats = batch["atomic_features"].to(device)
        mask = batch["mask"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(global_feats, atomic_feats, mask)

        # Loss calculation (MSE on log-transformed targets)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * targets.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Computes the loss (MSE) and the competition metric (Column-wise RMSLE).
    Since targets are already log(1+y), RMSLE is essentially sqrt(MSE) per column.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            global_feats = batch["global_features"].to(device)
            atomic_feats = batch["atomic_features"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(global_feats, atomic_feats, mask)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * targets.size(0)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    # Concatenate all predictions and targets
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate Column-wise RMSLE
    # Targets are already log1p transformed. Model predicts log1p values.
    # RMSLE_col = sqrt(mean((pred_col - target_col)^2))
    mse_per_col = np.mean((all_preds - all_targets) ** 2, axis=0)
    rmsle_per_col = np.sqrt(mse_per_col)
    mean_rmsle = np.mean(rmsle_per_col)

    return epoch_loss, mean_rmsle, rmsle_per_col


def run_training(max_samples=None, num_epochs=NUM_EPOCHS):
    """
    Orchestrates the entire training pipeline.
    """
    set_seed(SEED)
    setup_directories()

    print(f"Device: {DEVICE}")

    # 1. Load and Process Data
    print("Processing Training Data...")
    train_data = process_data(
        TRAIN_METADATA_PATH,
        TRAIN_CACHE_PATH,
        load_cached_data=True,
        max_samples=max_samples,
    )

    print("Processing Validation Data...")
    val_data = process_data(
        VAL_METADATA_PATH,
        VAL_CACHE_PATH,
        load_cached_data=True,
        max_samples=max_samples,
    )

    # 2. Fit Scaler
    print("Fitting Standard Scaler on Training Data...")
    scaler = StandardScaler()
    scaler.fit(
        torch.tensor(train_data["global_features"], dtype=torch.float32),
        torch.tensor(train_data["atomic_features_flat"], dtype=torch.float32),
    )

    # 3. Create Datasets and Loaders
    train_dataset = CrystalDataset(train_data, scaler=scaler)
    val_dataset = CrystalDataset(val_data, scaler=scaler)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False,
    )

    # 4. Initialize Model
    model = CADSTFModel().to(DEVICE)
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    # 5. Training Loop
    best_val_rmsle = float("inf")
    patience_counter = 0

    print("\nStarting Training...")
    print(
        f"{'Epoch':<6} | {'Train Loss':<12} | {'Val Loss':<12} | {'Val RMSLE':<12} | {'Time (s)':<8}"
    )
    print("-" * 60)

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_rmsle, val_rmsle_cols = validate(
            model, val_loader, criterion, DEVICE
        )

        scheduler.step(val_rmsle)

        elapsed = time.time() - start_time

        print(
            f"{epoch:<6} | {train_loss:<12.6f} | {val_loss:<12.6f} | {val_rmsle:<12.6f} | {elapsed:<8.2f}"
        )

        # Early Stopping and Model Checkpointing
        if val_rmsle < best_val_rmsle:
            best_val_rmsle = val_rmsle
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            # print(f"  -> New best model saved! (Formation: {val_rmsle_cols[0]:.4f}, Bandgap: {val_rmsle_cols[1]:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break

    print(f"\nTraining completed. Best Validation RMSLE: {best_val_rmsle:.6f}")
    return scaler


def generate_submission(scaler, max_samples=None):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("\nGenerating Submission...")

    # 1. Load Test Data
    test_data = process_data(
        TEST_METADATA_PATH,
        TEST_CACHE_PATH,
        load_cached_data=True,
        max_samples=max_samples,
    )

    # 2. Prepare Dataset and Loader
    test_dataset = CrystalDataset(test_data, scaler=scaler)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
    )

    # 3. Load Model
    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_SAVE_PATH}")

    model = CADSTFModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()

    # 4. Inference
    ids = []
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            global_feats = batch["global_features"].to(DEVICE)
            atomic_feats = batch["atomic_features"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            batch_ids = batch["ids"]

            outputs = model(global_feats, atomic_feats, mask)

            # Inverse transform: exp(y) - 1
            # Since outputs are log(1+y)
            preds_original_scale = torch.expm1(outputs)

            ids.extend(batch_ids.tolist())
            predictions.append(preds_original_scale.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)

    # 5. Create DataFrame and Save
    submission_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": predictions[:, 0],
            "bandgap_energy_ev": predictions[:, 1],
        }
    )

    # Sort by ID to match sample submission format usually
    submission_df.sort_values("id", inplace=True)

    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
    print(submission_df.head())
