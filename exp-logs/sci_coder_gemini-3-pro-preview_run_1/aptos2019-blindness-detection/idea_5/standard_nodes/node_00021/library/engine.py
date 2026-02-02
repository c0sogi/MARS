import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.model import RetinopathyModel
from library.data import get_dataloaders
from library.utils import compute_score, seed_everything


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, targets in loader:
        batch_size = images.size(0)
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(images)
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if Config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and Quadratic Weighted Kappa score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            batch_size = images.size(0)
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Decode Ordinal Predictions
            # 1. Sigmoid to get probabilities for each threshold
            probs = torch.sigmoid(logits)
            # 2. Sum probabilities to get continuous score (0 to 4)
            scores = probs.sum(dim=1)
            # 3. Round to nearest integer to get class label
            preds = scores.round().cpu().numpy().astype(int)

            # Decode Targets
            # Summing the binary ordinal vector recovers the integer class
            true_labels = targets.sum(dim=1).cpu().numpy().astype(int)

            all_preds.extend(preds)
            all_targets.extend(true_labels)

    val_loss = running_loss / dataset_size
    qwk = compute_score(all_targets, all_preds)

    return val_loss, qwk


def run_training():
    """
    Main training loop.
    Initializes model, optimizer, scheduler, and runs epochs.
    Saves the best model based on QWK.
    """
    seed_everything(Config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    # Get DataLoaders
    train_loader, val_loader, _ = get_dataloaders()

    # Initialize Model
    model = RetinopathyModel(pretrained=True)
    model.to(device)

    # Loss Function: BCEWithLogitsLoss for multi-label/ordinal targets
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.epochs, eta_min=Config.min_lr
    )

    best_qwk = -np.inf

    print("Starting training...")

    for epoch in range(Config.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_qwk = validate(model, val_loader, criterion, device)

        # Step the scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val QWK: {val_qwk}"
        )

        # Save Best Model
        if val_qwk > best_qwk:
            best_qwk = val_qwk
            torch.save(model.state_dict(), Config.model_save_path)
            print(f"New best model saved! QWK: {best_qwk}")

    print(f"Training finished. Best Validation QWK: {best_qwk}")


def generate_submission():
    """
    Loads the best model and generates predictions for the test set.
    Saves the result to submission.csv.
    """
    seed_everything(Config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Generating submission...")

    # Get Test DataLoader
    _, _, test_loader = get_dataloaders()

    # Load Model Structure
    model = RetinopathyModel(pretrained=False)

    # Load Weights
    if not os.path.exists(Config.model_save_path):
        raise FileNotFoundError(
            f"Model file not found at {Config.model_save_path}. Run training first."
        )

    model.load_state_dict(torch.load(Config.model_save_path, map_location=device))
    model.to(device)
    model.eval()

    id_codes = []
    predictions = []

    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)
            scores = probs.sum(dim=1)
            preds = scores.round().cpu().numpy().astype(int)

            # Clip to valid range [0, 4]
            preds = np.clip(preds, 0, 4)

            id_codes.extend(ids)
            predictions.extend(preds)

    # Create Submission DataFrame
    df_submission = pd.DataFrame({"id_code": id_codes, "diagnosis": predictions})

    # Save to CSV
    # Ensure directory exists
    os.makedirs(os.path.dirname(Config.submission_path), exist_ok=True)
    df_submission.to_csv(Config.submission_path, index=False)

    print(f"Submission saved to {Config.submission_path}")
    print(df_submission.head())
