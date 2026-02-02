import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from library.config import Config, seed_everything
from library.utils import get_device
from library.data import get_datasets
from library.model import VAMSHDNet


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        # Target needs to be (B, 1) to match output shape
        target = target.view(-1, 1)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns: avg_loss, auc_score
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            target = target.view(-1, 1)

            loss = criterion(output, target)
            running_loss += loss.item() * data.size(0)

            # Apply sigmoid to get probabilities for AUC
            probs = torch.sigmoid(output)

            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)

    # Calculate AUC
    # Handle edge case where batch might only have one class
    try:
        auc_score = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc_score = 0.5

    return avg_loss, auc_score


def predict_test_set(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for data in loader:
            # Test loader might return just data or (data, target) depending on implementation
            # Based on library/data.py, if y is None, it returns just img_tensor
            if isinstance(data, (tuple, list)):
                data = data[0]

            data = data.to(device)
            output = model(data)
            probs = torch.sigmoid(output)
            all_probs.extend(probs.cpu().numpy().flatten())

    return all_probs


def run_training():
    """
    Main execution function for training and submission generation.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Initializing datasets...")
    train_dataset, val_dataset, test_dataset = get_datasets(load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model & Optimization
    print("Initializing model...")
    model = VAMSHDNet().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 4. Training Loop with Early Stopping
    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc:.6f}"
        )

        # Early Stopping Logic (Maximizing AUC)
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with AUC: {best_auc:.6f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 5. Inference & Submission
    print("Loading best model for inference...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: No best model found. Using current model state.")

    print("Generating predictions on test set...")
    predictions = predict_test_set(model, test_loader, device)

    # Create submission DataFrame
    # Note: test_dataset.ids contains the BraTS21IDs in the same order as the loader (shuffle=False)
    submission_df = pd.DataFrame(
        {"BraTS21ID": test_dataset.ids, "MGMT_value": predictions}
    )

    # Ensure output directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")

    # Print head of submission for verification
    print(submission_df.head())
