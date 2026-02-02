import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import AverageMeter, calculate_roc_auc, set_seed
from library.data import get_dataloaders
from library.model import SpatialDifferenceCNN


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        loader: The training DataLoader.
        criterion: The loss function.
        optimizer: The optimizer.
        device: The device to train on.

    Returns:
        tuple: (average_loss, auc_score)
    """
    model.train()
    losses = AverageMeter()

    all_targets = []
    all_preds = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

        # Collect for AUC calculation
        probs = torch.sigmoid(logits)
        all_targets.append(targets.detach().cpu())
        all_preds.append(probs.detach().cpu())

    all_targets = torch.cat(all_targets).numpy()
    all_preds = torch.cat(all_preds).numpy()

    epoch_auc = calculate_roc_auc(all_targets, all_preds)

    return losses.avg, epoch_auc


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: The validation DataLoader.
        criterion: The loss function.
        device: The device to evaluate on.

    Returns:
        tuple: (average_loss, auc_score)
    """
    model.eval()
    losses = AverageMeter()

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, targets)

            losses.update(loss.item(), images.size(0))

            probs = torch.sigmoid(logits)
            all_targets.append(targets.cpu())
            all_preds.append(probs.cpu())

    all_targets = torch.cat(all_targets).numpy()
    all_preds = torch.cat(all_preds).numpy()

    epoch_auc = calculate_roc_auc(all_targets, all_preds)

    return losses.avg, epoch_auc


def run_training(debug=False, epochs=Config.MAX_EPOCHS, patience=Config.PATIENCE):
    """
    Main training routine.

    Args:
        debug (bool): Whether to run in debug mode (smaller dataset).
        epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Starting training on device: {device}")

    # Get DataLoaders
    train_loader, val_loader, _ = get_dataloaders(debug=debug)

    # Initialize Model, Criterion, Optimizer
    model = SpatialDifferenceCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    best_val_auc = 0.0
    patience_counter = 0
    best_epoch = 0

    # Ensure working directory exists for model saving
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    for epoch in range(epochs):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Print metrics with full precision (no formatting)
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss} | Train AUC: {train_auc}")
        print(f"  Val Loss:   {val_loss} | Val AUC:   {val_auc}")

        # Early Stopping & Checkpointing
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  New best model saved to {Config.MODEL_SAVE_PATH}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val AUC: {best_val_auc} at epoch {best_epoch+1}")


def predict_and_submit(debug=False):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        debug (bool): Whether to run in debug mode.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Get Test DataLoader
    _, _, test_loader = get_dataloaders(debug=debug)

    # Load Model
    model = SpatialDifferenceCNN().to(device)
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model file not found at {Config.MODEL_SAVE_PATH}. Train first."
        )

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    print("Starting inference on test set...")

    ids = []
    predictions = []

    with torch.no_grad():
        for images, sample_ids in test_loader:
            images = images.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)

            # Flatten predictions to 1D array
            probs_np = probs.cpu().numpy().flatten()

            ids.extend(sample_ids)
            predictions.extend(probs_np)

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"id": ids, "target": predictions})

    # Save Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
