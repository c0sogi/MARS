import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from library.config import Config, set_seed
from library.utils import get_logger, get_device
from library.data_loader import get_dataloaders
from library.model import MGMT25DModel

logger = get_logger()


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Store predictions and targets for AUC calculation
        all_targets.append(targets.detach().cpu().numpy())
        all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        # Handle edge cases with single class in batch/epoch
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def validate_epoch(model, loader, criterion, device):
    """
    Executes validation on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = criterion(logits, targets)

            running_loss += loss.item() * inputs.size(0)

            all_targets.append(targets.detach().cpu().numpy())
            all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def run_training():
    """
    Main training pipeline with Early Stopping.
    """
    set_seed(Config.SEED)
    device = get_device()
    logger.info(f"Starting training on device: {device}")

    # Load Data
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Initialize Model
    model = MGMT25DModel().to(device)

    # Optimizer (Adam, no weight decay) & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    # Training State
    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    for epoch in range(Config.EPOCHS):
        train_loss, train_auc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_auc = validate_epoch(model, val_loader, criterion, device)

        # Print full precision metrics
        logger.info(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.15f} | Train AUC: {train_auc:.15f} | "
            f"Val Loss: {val_loss:.15f} | Val AUC: {val_auc:.15f}"
        )

        # Early Stopping Logic
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved with AUC: {best_auc:.15f}")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break

    return best_model_path, test_loader


def generate_submission(model_path, test_loader):
    """
    Generates predictions for the test set and saves submission.csv.
    """
    device = get_device()
    model = MGMT25DModel().to(device)

    if not os.path.exists(model_path):
        logger.error(
            f"Model file not found at {model_path}. Cannot generate submission."
        )
        return

    logger.info(f"Loading best model from {model_path} for inference...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    predictions = []
    ids = []

    with torch.no_grad():
        for inputs, pids in test_loader:
            inputs = inputs.to(device)

            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            # Handle pids whether they are tensors or tuples
            if isinstance(pids, torch.Tensor):
                pids = pids.numpy()

            predictions.extend(probs)
            ids.extend(pids)

    # Create Submission DataFrame
    df = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": predictions})

    # Ensure ID format matches sample submission (integer)
    df["BraTS21ID"] = df["BraTS21ID"].astype(int)

    # Save
    df.to_csv(Config.SUBMISSION_FILE, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_FILE}")


def main():
    best_model_path, test_loader = run_training()
    generate_submission(best_model_path, test_loader)


if __name__ == "__main__":
    main()
