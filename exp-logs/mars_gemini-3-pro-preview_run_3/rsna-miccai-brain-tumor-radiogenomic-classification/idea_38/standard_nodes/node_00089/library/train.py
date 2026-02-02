import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.data import get_dataloader
from library.model import S3DNet


def set_seed(seed=Config.SEED):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for batch in loader:
        # Unpack batch
        even = batch["even"].to(device)
        odd = batch["odd"].to(device)
        targets = batch["target"].to(device).unsqueeze(1)  # Shape (B, 1)

        optimizer.zero_grad()

        # Forward pass
        logits = model(even, odd)
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * even.size(0)
        count += even.size(0)

    return running_loss / count if count > 0 else 0.0


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC score.
    """
    model.eval()
    running_loss = 0.0
    count = 0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            even = batch["even"].to(device)
            odd = batch["odd"].to(device)
            targets = batch["target"].to(device).unsqueeze(1)

            logits = model(even, odd)
            loss = criterion(logits, targets)

            probs = torch.sigmoid(logits)

            running_loss += loss.item() * even.size(0)
            count += even.size(0)

            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    avg_loss = running_loss / count if count > 0 else 0.0

    # Concatenate for metric calculation
    if len(all_targets) > 0:
        all_targets = np.concatenate(all_targets)
        all_probs = np.concatenate(all_probs)

        # Handle edge case where only one class is present in the batch/set
        if len(np.unique(all_targets)) > 1:
            auc = roc_auc_score(all_targets, all_probs)
        else:
            auc = 0.5
    else:
        auc = 0.5

    return avg_loss, auc


def run_training():
    """
    Main function to run the training pipeline with Early Stopping.
    """
    set_seed()
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Initialize DataLoaders
    # We use load_cached_data=True to leverage the caching mechanism implemented in library.utils
    train_loader = get_dataloader("train", load_cached_data=True)
    val_loader = get_dataloader("val", load_cached_data=True)

    # Initialize Model
    model = S3DNet()
    model.to(device)

    # Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    # Early Stopping Setup
    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    # Ensure directory exists
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    print("Starting training...")

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}/{Config.NUM_EPOCHS} - "
            f"Train Loss: {train_loss} - "
            f"Val Loss: {val_loss} - "
            f"Val AUC: {val_auc}"
        )

        # Check Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with AUC: {best_auc}")
        else:
            patience_counter += 1
            print(
                f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation AUC: {best_auc}")

    # Load best model for return
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    return model
