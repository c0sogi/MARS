import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score
from library.utils import seed_everything, get_device
from library.data_loader import get_dataloaders
from library.model import DSSVNet


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, ((even_stream, odd_stream), targets) in enumerate(loader):
        # Move data to device
        even_stream = even_stream.to(device)
        odd_stream = odd_stream.to(device)
        targets = targets.to(device).view(-1, 1)  # Ensure target shape matches logits

        optimizer.zero_grad()

        # Forward pass
        logits = model(even_stream, odd_stream)

        # Calculate loss
        loss = criterion(logits, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    return avg_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC score.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for (even_stream, odd_stream), targets in loader:
            even_stream = even_stream.to(device)
            odd_stream = odd_stream.to(device)
            targets = targets.to(device).view(-1, 1)

            logits = model(even_stream, odd_stream)
            loss = criterion(logits, targets)

            running_loss += loss.item()

            # Apply sigmoid to get probabilities for AUC calculation
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(targets.cpu().numpy())

    avg_loss = running_loss / len(loader)

    # Handle potential edge cases where only one class is present in the batch/split
    try:
        auc_score = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc_score = 0.5

    return avg_loss, auc_score


def run_training(
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-4,
    input_dir: str = "./input",
    cache_dir: str = "./working/idea_35/",
    limit_size: int = None,
    seed: int = 42,
):
    """
    Orchestrates the training process.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size for dataloaders.
        lr: Learning rate for Adam optimizer.
        input_dir: Root directory of input data.
        cache_dir: Directory to store cached data and model checkpoints.
        limit_size: If set, limits dataset size for debugging.
        seed: Random seed for reproducibility.

    Returns:
        best_model_path: Path to the saved best model checkpoint.
    """
    # Setup
    os.makedirs(cache_dir, exist_ok=True)
    seed_everything(seed)
    device = get_device()

    print(f"Starting training on device: {device}")

    # Load Data
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=batch_size,
        input_dir=input_dir,
        cache_dir=cache_dir,
        load_cached_data=True,
        limit_size=limit_size,
        seed=seed,
    )

    # Initialize Model
    model = DSSVNet(pretrained=True)
    model = model.to(device)

    # Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # Tracking
    best_auc = 0.0
    best_model_path = os.path.join(cache_dir, "best_model.pth")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with AUC: {best_auc}")

    print(f"Training complete. Best AUC: {best_auc}")
    return best_model_path
