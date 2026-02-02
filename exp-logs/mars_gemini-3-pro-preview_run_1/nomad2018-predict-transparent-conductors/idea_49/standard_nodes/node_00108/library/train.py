import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.data import CrystalDataset, collate_sparse_batch
from library.model import GBAMSDSModel


# ==========================================
# Reproducibility
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==========================================
# Training & Validation Functions
# ==========================================
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        # Move batch data to device
        atomic_features = batch["atomic_features"].to(device)
        global_features = batch["global_features"].to(device)
        batch_indices = batch["batch_indices"].to(device)
        targets = batch["targets"].to(device)

        # Prepare input dict
        inputs = {
            "atomic_features": atomic_features,
            "global_features": global_features,
            "batch_indices": batch_indices,
        }

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)

    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    # For reporting column-wise metrics
    total_loss_formation = 0.0
    total_loss_bandgap = 0.0

    with torch.no_grad():
        for batch in loader:
            atomic_features = batch["atomic_features"].to(device)
            global_features = batch["global_features"].to(device)
            batch_indices = batch["batch_indices"].to(device)
            targets = batch["targets"].to(device)

            inputs = {
                "atomic_features": atomic_features,
                "global_features": global_features,
                "batch_indices": batch_indices,
            }

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size

            # Calculate individual column MSEs for reporting
            # outputs and targets are (B, 2)
            # Column 0: Formation Energy, Column 1: Bandgap Energy
            mse_per_col = torch.mean((outputs - targets) ** 2, dim=0)
            total_loss_formation += mse_per_col[0].item() * batch_size
            total_loss_bandgap += mse_per_col[1].item() * batch_size

    avg_loss = total_loss / len(loader.dataset)
    avg_loss_form = total_loss_formation / len(loader.dataset)
    avg_loss_band = total_loss_bandgap / len(loader.dataset)

    # Since targets are log-transformed, sqrt(MSE) approximates RMSLE
    rmsle_form = np.sqrt(avg_loss_form)
    rmsle_band = np.sqrt(avg_loss_band)

    return avg_loss, rmsle_form, rmsle_band


# ==========================================
# Main Training Routine
# ==========================================
def run_training(debug_mode=False):
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Prepare Datasets
    print("Initializing Training Dataset...")
    train_dataset = CrystalDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        cache_file=Config.TRAIN_CACHE_FILE,
        fit_scalers=True,
        transform_targets=True,
        load_cached_data=True,
    )

    print("Initializing Validation Dataset...")
    # Pass scalers from training set to validation set
    val_dataset = CrystalDataset(
        metadata_path=Config.VAL_METADATA_PATH,
        cache_file=Config.VAL_CACHE_FILE,
        scalers=(train_dataset.a_scaler, train_dataset.g_scaler),
        fit_scalers=False,
        transform_targets=True,
        load_cached_data=True,
    )

    # 2. DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_sparse_batch,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_sparse_batch,
        num_workers=2,
    )

    # 3. Model Setup
    model = GBAMSDSModel().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.SCHEDULER_MIN_LR,
        verbose=True,
    )
    criterion = nn.MSELoss()

    # 4. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")
    start_time = time.time()

    epochs = 5 if debug_mode else Config.EPOCHS

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_rmsle_form, val_rmsle_band = validate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)

        # Logging
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val RMSLE (Form): {val_rmsle_form:.6f} | "
            f"Val RMSLE (Band): {val_rmsle_band:.6f}"
        )

        # Checkpointing & Early Stopping
        if val_loss < best_val_loss - Config.MIN_DELTA:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            # print(f"  -> Model saved. Best Val Loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training finished in {time.time() - start_time:.2f} seconds.")
    print(f"Best Validation Loss: {best_val_loss:.6f}")


# ==========================================
# Submission Generation
# ==========================================
def generate_submission(debug_mode=False):
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Load Test Data
    # We need to reuse the scalers used during training.
    # The CrystalDataset logic will look for saved scalers in the working dir if not provided directly.
    # Since we ran training, scalers.npz should exist.
    print("Initializing Test Dataset...")
    test_dataset = CrystalDataset(
        metadata_path=Config.TEST_METADATA_PATH,
        cache_file=Config.TEST_CACHE_FILE,
        fit_scalers=False,
        transform_targets=False,  # No targets in test
        load_cached_data=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_sparse_batch,
        num_workers=2,
    )

    # 2. Load Model
    model = GBAMSDSModel().to(device)
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
        print("Loaded best model checkpoint.")
    else:
        print(
            "Warning: No model checkpoint found. Using untrained model (random predictions)."
        )

    model.eval()

    # 3. Inference
    all_ids = []
    all_preds = []

    print("Running inference...")
    with torch.no_grad():
        for batch in test_loader:
            atomic_features = batch["atomic_features"].to(device)
            global_features = batch["global_features"].to(device)
            batch_indices = batch["batch_indices"].to(device)
            ids = batch["ids"]

            inputs = {
                "atomic_features": atomic_features,
                "global_features": global_features,
                "batch_indices": batch_indices,
            }

            # Forward pass (log space)
            outputs_log = model(inputs)

            # Inverse transform: exp(x) - 1
            outputs_original = torch.expm1(outputs_log)

            all_ids.append(ids.cpu().numpy())
            all_preds.append(outputs_original.cpu().numpy())

    # 4. Format Submission
    all_ids = np.concatenate(all_ids)
    all_preds = np.concatenate(all_preds)

    submission_df = pd.DataFrame(
        {
            "id": all_ids,
            "formation_energy_ev_natom": all_preds[:, 0],
            "bandgap_energy_ev": all_preds[:, 1],
        }
    )

    # Ensure columns are in correct order
    submission_df = submission_df[
        ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
    ]

    # Sort by ID just in case
    submission_df.sort_values("id", inplace=True)

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission_df.head())


if __name__ == "__main__":
    # Run full pipeline
    run_training()
    generate_submission()
