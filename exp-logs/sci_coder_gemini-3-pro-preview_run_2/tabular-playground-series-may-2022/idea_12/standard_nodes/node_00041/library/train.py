import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import seed_everything, get_device
from library.data_loader import get_dataloaders
from library.model import GlobalContextTransformerResFunnel


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.

    Args:
        model: The PyTorch model.
        loader: The training DataLoader.
        optimizer: The optimizer.
        criterion: The loss function.
        device: The device to run on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        cont = batch["continuous"].to(device)
        seq = batch["sequence"].to(device)
        target = batch["target"].to(device).unsqueeze(1)

        optimizer.zero_grad()
        output = model(cont, seq)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: The validation DataLoader.
        criterion: The loss function.
        device: The device to run on.

    Returns:
        tuple: (Average validation loss, Validation AUC score)
    """
    model.eval()
    running_loss = 0.0
    preds = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            cont = batch["continuous"].to(device)
            seq = batch["sequence"].to(device)
            target = batch["target"].to(device).unsqueeze(1)

            output = model(cont, seq)
            loss = criterion(output, target)
            running_loss += loss.item()

            preds.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())

    avg_loss = running_loss / len(loader)
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    try:
        auc = roc_auc_score(targets, preds)
    except ValueError:
        # Fallback if only one class is present in batch (rare/debug cases)
        auc = 0.5

    return avg_loss, auc


def run_training(epochs=Config.EPOCHS, debug=False):
    """
    Main training loop with Early Stopping and Scheduler.

    Args:
        epochs (int): Number of training epochs.
        debug (bool): If True, runs with a small subset of data.

    Returns:
        float: Best Validation AUC achieved.
    """
    # Configure Debug Mode
    if debug:
        Config.DEBUG = True

    seed_everything()
    device = get_device()

    # Load Data
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=True)

    # Model Initialization
    model = GlobalContextTransformerResFunnel().to(device)

    # Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.SCHEDULER_STEP_SIZE, gamma=Config.SCHEDULER_GAMMA
    )

    criterion = nn.BCELoss()

    # Tracking
    best_val_auc = 0.0
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Print metrics with full precision (no rounding) as required
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc} | LR: {current_lr}"
        )

        # Early Stopping & Checkpointing
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"--> Best model saved! AUC: {best_val_auc}")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    return best_val_auc


def generate_submission(debug=False):
    """
    Generates predictions for the test set using the best saved model.

    Args:
        debug (bool): If True, runs with a small subset of data.
    """
    if debug:
        Config.DEBUG = True

    device = get_device()

    # Load test data
    _, _, test_loader = get_dataloaders(load_cached_data=True)

    model = GlobalContextTransformerResFunnel().to(device)

    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_SAVE_PATH}")

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    print("Generating predictions on test set...")

    ids_list = []
    preds_list = []

    with torch.no_grad():
        for batch in test_loader:
            cont = batch["continuous"].to(device)
            seq = batch["sequence"].to(device)
            ids = batch["id"]

            output = model(cont, seq)

            ids_list.append(ids.numpy())
            preds_list.append(output.cpu().numpy())

    all_ids = np.concatenate(ids_list)
    all_preds = np.concatenate(preds_list).flatten()

    df_sub = pd.DataFrame({"id": all_ids, "target": all_preds})

    # Ensure directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved successfully to {Config.SUBMISSION_PATH}")
