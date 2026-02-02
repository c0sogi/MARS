import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from library.config import Config
from library.data import get_dataloaders
from library.model import SRACGN
from library.utils import get_scaler, compute_metric


def set_seed(seed):
    """
    Sets the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_epoch(model, loader, optimizer, criterion, scaler, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0.0
    num_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch)

        # Scale targets for loss calculation
        targets = batch.y
        targets_np = targets.cpu().numpy()
        scaled_targets_np = scaler.transform(targets_np)
        scaled_targets = torch.tensor(
            scaled_targets_np, dtype=torch.float32, device=device
        )

        # Compute loss
        loss = criterion(outputs, scaled_targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        num_graphs += batch.num_graphs

    return total_loss / num_graphs


def evaluate(model, loader, criterion, scaler, device):
    """
    Evaluates the model on the given loader.
    Returns average loss (scaled) and RMSLE (original scale).
    """
    model.eval()
    total_loss = 0.0
    num_graphs = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)

            # Scale targets for loss calculation
            targets = batch.y
            targets_np = targets.cpu().numpy()
            scaled_targets_np = scaler.transform(targets_np)
            scaled_targets = torch.tensor(
                scaled_targets_np, dtype=torch.float32, device=device
            )

            # Compute loss
            loss = criterion(outputs, scaled_targets)
            total_loss += loss.item() * batch.num_graphs
            num_graphs += batch.num_graphs

            # Inverse transform predictions for metric calculation
            preds_np = scaler.inverse_transform(outputs.cpu().numpy())

            all_preds.append(preds_np)
            all_targets.append(targets_np)

    avg_loss = total_loss / num_graphs

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    rmsle = compute_metric(all_targets, all_preds)

    return avg_loss, rmsle


def run_training(
    epochs=Config.NUM_EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    patience=Config.EARLY_STOPPING_PATIENCE,
    load_cached_data=True,
):
    """
    Main function to run the training pipeline.
    """
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running training on device: {device}")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Load Data
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=load_cached_data)

    # Prepare Scaler
    # We need to fit the scaler on the training data.
    # Collect all training targets to fit the scaler.
    print("Preparing target scaler...")
    all_train_targets = []
    for data in train_loader.dataset:
        all_train_targets.append(data.y.numpy())
    all_train_targets = np.concatenate(all_train_targets, axis=0)

    scaler = get_scaler(all_train_targets, load_cached_data=load_cached_data)

    # Initialize Model
    model = SRACGN().to(device)

    # Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.SCHEDULER_MIN_LR,
    )

    # Loss Function (Mean Squared Error)
    criterion = nn.MSELoss()

    # Training Loop
    best_val_loss = float("inf")
    early_stopping_counter = 0

    print("Starting training loop...")
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device
        )
        val_loss, val_rmsle = evaluate(model, val_loader, criterion, scaler, device)

        # Update Scheduler
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch}: Train Loss: {train_loss}, Val Loss: {val_loss}, Val RMSLE: {val_rmsle}"
        )

        # Checkpoint and Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
            # print(f"New best model saved to {Config.MODEL_CHECKPOINT_PATH}")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    print("Training process completed.")
