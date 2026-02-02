import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from library.config import ModelConfig, seed_everything
from library.utils import save_checkpoint, load_checkpoint
from library.dataset import get_dataloaders
from library.model import IIResFunnelGLU


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        loader: DataLoader for training data.
        optimizer: The optimizer.
        criterion: The loss function.
        device: The device to run training on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Move data to device
        cont = batch["cont"].to(device)
        cat = batch["cat"].to(device)
        target = batch["target"].to(device).view(-1, 1)

        optimizer.zero_grad()

        # Forward pass
        logits = model(cont, cat)
        loss = criterion(logits, target)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: DataLoader for validation data.
        criterion: The loss function.
        device: The device to run evaluation on.

    Returns:
        tuple: (average_loss, auc_score)
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            cont = batch["cont"].to(device)
            cat = batch["cat"].to(device)
            target = batch["target"].to(device).view(-1, 1)

            logits = model(cont, cat)
            loss = criterion(logits, target)

            running_loss += loss.item()

            # Apply sigmoid to get probabilities for AUC calculation
            probs = torch.sigmoid(logits)

            all_targets.append(target.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    auc = roc_auc_score(all_targets, all_preds)

    return running_loss / len(loader), auc


def predict(model, loader, device):
    """
    Generates predictions for the test set.

    Args:
        model: The PyTorch model.
        loader: DataLoader for test data.
        device: The device to run prediction on.

    Returns:
        np.ndarray: Flattened array of predicted probabilities.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            cont = batch["cont"].to(device)
            cat = batch["cat"].to(device)

            logits = model(cont, cat)
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())

    return np.concatenate(all_preds).flatten()


def run_pipeline():
    """
    Executes the full training and submission pipeline.
    """
    # 1. Setup
    config = ModelConfig
    seed_everything(config.SEED)
    device = torch.device(config.DEVICE)

    # 2. Data Loading
    # get_dataloaders handles the caching logic internally via prepare_data
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=True, debug=config.DEBUG
    )

    # 3. Model Initialization
    model = IIResFunnelGLU().to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    print("Starting training...")
    best_auc = 0.0
    patience = 0

    for epoch in range(config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val AUC: {val_auc}"
        )

        # Checkpoint based on AUC
        if val_auc > best_auc:
            best_auc = val_auc
            patience = 0
            save_checkpoint(
                model, optimizer, None, epoch, val_auc, config.MODEL_SAVE_PATH
            )
        else:
            patience += 1
            if patience >= config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {patience} epochs.")
                break

    # 6. Submission Generation
    print("Generating submission...")

    # Load best model
    checkpoint = load_checkpoint(config.MODEL_SAVE_PATH, model, device=device)
    print(
        f"Loaded best model from epoch {checkpoint['epoch']} with AUC {checkpoint['metric']}"
    )

    # Predict
    predictions = predict(model, test_loader, device)

    # Format submission
    submission = pd.DataFrame({"id": test_ids, "target": predictions})

    # Save
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)
    submission.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
