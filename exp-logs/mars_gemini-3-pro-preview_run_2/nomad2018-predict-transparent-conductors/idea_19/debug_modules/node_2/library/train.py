import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os

from library.config import Config
from library.utils import set_seed, get_device
from library.data import get_dataloaders
from library.model import SR_CGN_DP


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        loader: DataLoader for training data.
        optimizer: The optimizer.
        criterion: The loss function.
        device: The device to run on.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_samples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch)

        # Targets are already scaled in the dataset if using the provided CrystalDataset
        targets = batch.y

        loss = criterion(outputs, targets)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        num_samples += batch.num_graphs

    return total_loss / num_samples


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: DataLoader for validation data.
        criterion: The loss function.
        device: The device to run on.

    Returns:
        Average validation loss.
    """
    model.eval()
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            outputs = model(batch)
            targets = batch.y

            loss = criterion(outputs, targets)

            total_loss += loss.item() * batch.num_graphs
            num_samples += batch.num_graphs

    return total_loss / num_samples


def generate_submission(model, test_loader, scaler, device, output_path):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model: The trained PyTorch model.
        test_loader: DataLoader for test data.
        scaler: The fitted TargetScaler to inverse transform predictions.
        device: The device to run on.
        output_path: Path to save the submission CSV.
    """
    model.eval()
    ids = []
    preds = []

    print("Generating submission...")
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outputs = model(batch)

            # Move to CPU and numpy
            outputs_np = outputs.cpu().numpy()

            # Inverse transform to get original units (eV)
            if scaler:
                outputs_np = scaler.inverse_transform(outputs_np)

            preds.append(outputs_np)
            ids.extend(batch.id.cpu().numpy())

    # Concatenate predictions
    preds = np.concatenate(preds, axis=0)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": preds[:, 0],
            "bandgap_energy_ev": preds[:, 1],
        }
    )

    # Sort by ID
    df = df.sort_values("id")

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(
    data_sample_size=Config.DATA_SAMPLE_SIZE,
    num_epochs=Config.NUM_EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    patience=Config.EARLY_STOPPING_PATIENCE,
    load_cached_data=True,
):
    """
    Main function to run the training pipeline.

    Args:
        data_sample_size: Number of samples to use (for debugging). None for full dataset.
        num_epochs: Maximum number of epochs.
        batch_size: Batch size.
        learning_rate: Initial learning rate.
        weight_decay: Weight decay for optimizer.
        patience: Patience for early stopping.
        load_cached_data: Whether to load pre-processed graphs from cache.

    Returns:
        Best validation loss achieved.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = get_device()
    print(f"Using device: {device}")

    # 2. Data
    train_loader, val_loader, test_loader, scaler = get_dataloaders(
        batch_size=batch_size,
        num_workers=Config.NUM_WORKERS,
        data_sample_size=data_sample_size,
        load_cached_data=load_cached_data,
    )

    # 3. Model
    model = SR_CGN_DP(
        node_dim=Config.ATOM_EMBEDDING_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout_rate=Config.DROPOUT_RATE,
        rbf_bins=Config.RBF_BINS,
        rbf_min=Config.RBF_MIN,
        rbf_max=Config.RBF_MAX,
    ).to(device)

    # 4. Optimizer & Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Scheduler: ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # MSE Loss on scaled targets
    criterion = nn.MSELoss()

    # 5. Training Loop
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print("Starting training...")
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        epoch_time = time.time() - start_time

        # Print metrics with full precision
        print(
            f"Epoch {epoch}/{num_epochs} | Time: {epoch_time:.2f}s | Train Loss: {train_loss} | Val Loss: {val_loss}"
        )

        # Checkpoint & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved! Val Loss: {val_loss}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    # 6. Generate Submission
    print("Loading best model for submission...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: Best model checkpoint not found. Using current model state.")

    generate_submission(model, test_loader, scaler, device, Config.SUBMISSION_PATH)

    return best_val_loss
