import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed, Standardizer
from library.data import get_dataloaders
from library.model import HASCNet


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): The training data loader.
        optimizer (torch.optim.Optimizer): The optimizer.
        criterion (nn.Module): The loss function.
        device (torch.device): The device to run on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_graphs = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, data.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        num_graphs += data.num_graphs

    return total_loss / num_graphs


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): The validation data loader.
        criterion (nn.Module): The loss function.
        device (torch.device): The device to run on.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    total_loss = 0.0
    num_graphs = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data.y)
            total_loss += loss.item() * data.num_graphs
            num_graphs += data.num_graphs

    return total_loss / num_graphs


def run_training(load_cached_data=True):
    """
    Main training loop for HASC-Net.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from cache.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Prepare Data
    train_loader, val_loader, _, _ = get_dataloaders(load_cached_data=load_cached_data)

    # 2. Initialize Model
    model = HASCNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.MSELoss()

    # 3. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    checkpoint_dir = os.path.join(Config.WORKING_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

    print("Starting training...")
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss} | Val Loss: {val_loss}")

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    print(f"Training complete. Best Validation Loss: {best_val_loss}")


def generate_submission(load_cached_data=True):
    """
    Generates predictions for the test set using the best trained model.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from cache.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Prepare Data (Test set)
    # We need the test loader. get_dataloaders handles the scaling logic.
    _, _, test_loader, _ = get_dataloaders(load_cached_data=load_cached_data)

    # Load the target scaler
    # We prefer loading the one saved during training to ensure consistency
    target_scaler = Standardizer(device=device)
    scaler_path = os.path.join(Config.WORKING_DIR, "target_scaler.npz")

    if os.path.exists(scaler_path):
        target_scaler.load(scaler_path)
    else:
        # Fallback: re-fit if file is missing (e.g. if training wasn't run in this session)
        print("Scaler file not found, re-fitting from data...")
        _, _, _, target_scaler_fit = get_dataloaders(load_cached_data=load_cached_data)
        target_scaler = target_scaler_fit

    # 2. Load Model
    model = HASCNet().to(device)
    checkpoint_path = os.path.join(Config.WORKING_DIR, "checkpoints", "best_model.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}. Run training first."
        )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    print("Generating predictions...")
    ids = []
    predictions = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)

            # Inverse transform predictions to original scale
            # Note: target_scaler handles device movement internally if needed
            out_inv = target_scaler.inverse_transform(out)

            ids.extend(data.material_id)
            predictions.append(out_inv.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)

    # 3. Save Submission
    submission_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": predictions[:, 0],
            "bandgap_energy_ev": predictions[:, 1],
        }
    )

    # Ensure submission directory exists
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, "submission.csv")

    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
