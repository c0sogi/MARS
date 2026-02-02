import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import time
from torch.utils.data import DataLoader

from library.config import Config
from library.model import CGCNN
from library.dataset import CrystalGraphDataset, collate_graphs
from library.data_utils import StandardScaler


def calculate_rmsle(preds, targets):
    """
    Calculates Column-wise Root Mean Squared Logarithmic Error.
    preds: (N, 2) tensor or array
    targets: (N, 2) tensor or array
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Ensure non-negative for log input (clipping at 0)
    preds = np.maximum(preds, 0)
    targets = np.maximum(targets, 0)

    log_preds = np.log1p(preds)
    log_targets = np.log1p(targets)

    squared_errors = (log_preds - log_targets) ** 2
    mse = np.mean(squared_errors, axis=0)
    rmsle_per_col = np.sqrt(mse)

    return np.mean(rmsle_per_col)


def train_one_epoch(model, loader, optimizer, criterion, device):
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

        # Loss calculation (on standardized targets)
        loss = criterion(preds, batch.y)

        # Backpropagation
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        running_loss += loss.item() * batch.num_graphs

    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device, target_scaler):
    """
    Evaluates the model on a validation or test set.
    Returns average loss (scaled) and RMSLE (original scale).
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds = model(batch)

            # Loss on scaled values
            if batch.y is not None:
                loss = criterion(preds, batch.y)
                running_loss += loss.item() * batch.num_graphs

                # Inverse transform targets for metric calculation
                targets_real = target_scaler.inverse_transform(batch.y)
                all_targets.append(targets_real.cpu())

            # Inverse transform predictions
            preds_real = target_scaler.inverse_transform(preds)
            all_preds.append(preds_real.cpu())

    all_preds = torch.cat(all_preds, dim=0)

    epoch_loss = 0.0
    epoch_rmsle = 0.0

    if len(all_targets) > 0:
        all_targets = torch.cat(all_targets, dim=0)
        epoch_loss = running_loss / len(loader.dataset)
        epoch_rmsle = calculate_rmsle(all_preds, all_targets)

    return epoch_loss, epoch_rmsle


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, checkpoint_path="best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.checkpoint_path)


def save_scalers(global_scaler, target_scaler, save_dir):
    """Saves scaler states to disk."""
    torch.save(global_scaler.state_dict(), os.path.join(save_dir, "global_scaler.pth"))
    torch.save(target_scaler.state_dict(), os.path.join(save_dir, "target_scaler.pth"))


def load_scalers(save_dir):
    """Loads scaler states from disk."""
    g_scaler = StandardScaler()
    t_scaler = StandardScaler()
    g_scaler.load_state_dict(torch.load(os.path.join(save_dir, "global_scaler.pth")))
    t_scaler.load_state_dict(torch.load(os.path.join(save_dir, "target_scaler.pth")))
    return g_scaler, t_scaler


def run_training(sample_size=None, epochs=Config.NUM_EPOCHS):
    """
    Orchestrates the training pipeline.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Config.setup_directories()

    # 1. Prepare Data
    print("Preparing Training Data...")
    train_dataset = CrystalGraphDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        cache_prefix="train",
        load_cached_data=True,
        global_scaler=StandardScaler(),
        target_scaler=StandardScaler(),
        fit_scalers=True,
        sample_limit=sample_size,
    )

    # Save scalers for inference
    save_scalers(
        train_dataset.global_scaler, train_dataset.target_scaler, Config.CHECKPOINT_DIR
    )

    print("Preparing Validation Data...")
    val_dataset = CrystalGraphDataset(
        metadata_path=Config.VAL_METADATA_PATH,
        cache_prefix="val",
        load_cached_data=True,
        global_scaler=train_dataset.global_scaler,
        target_scaler=train_dataset.target_scaler,
        fit_scalers=False,
        sample_limit=sample_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_graphs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_graphs,
    )

    # 2. Setup Model
    model = CGCNN().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.MSELoss()

    # 3. Training Loop
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    early_stopping = EarlyStopping(
        patience=Config.PATIENCE, checkpoint_path=checkpoint_path
    )

    print("Starting training...")
    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_rmsle = evaluate(
            model, val_loader, criterion, device, train_dataset.target_scaler
        )

        epoch_time = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val RMSLE: {val_rmsle} | "
            f"Time: {epoch_time:.2f}s"
        )

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val Loss: {early_stopping.best_loss:.6f}")
    return train_dataset.global_scaler, train_dataset.target_scaler


def generate_submission(global_scaler=None, target_scaler=None, sample_size=None):
    """
    Generates predictions for the test set and saves to CSV.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Generating submission...")

    # Load scalers if not provided
    if global_scaler is None or target_scaler is None:
        try:
            global_scaler, target_scaler = load_scalers(Config.CHECKPOINT_DIR)
        except FileNotFoundError:
            print("Scalers not found. Ensure training has been run.")
            return

    # Load Test Data
    test_dataset = CrystalGraphDataset(
        metadata_path=Config.TEST_METADATA_PATH,
        cache_prefix="test",
        load_cached_data=True,
        global_scaler=global_scaler,
        target_scaler=None,  # No targets in test
        fit_scalers=False,
        sample_limit=sample_size,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_graphs,
    )

    # Load Model
    model = CGCNN().to(device)
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    ids = []
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            preds_scaled = model(batch)
            preds_real = target_scaler.inverse_transform(preds_scaled)

            # Ensure non-negative predictions for physical realism (energies >= 0)
            preds_real = torch.clamp(preds_real, min=0.0)

            ids.extend(batch.material_id.cpu().numpy())
            predictions.extend(preds_real.cpu().numpy())

    # Create DataFrame
    predictions = np.array(predictions)
    submission_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": predictions[:, 0],
            "bandgap_energy_ev": predictions[:, 1],
        }
    )

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
