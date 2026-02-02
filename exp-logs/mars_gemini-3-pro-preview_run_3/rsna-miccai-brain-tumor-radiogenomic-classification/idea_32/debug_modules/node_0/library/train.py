import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from library.config import (
    WORKING_DIR,
    DEVICE,
    SEED,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    PATIENCE,
)
from library.utils import seed_everything
from library.data import load_processed_data, BraTSDataset
from library.model import HRLNNet


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(inputs)
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Store predictions and targets for AUC calculation
        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        # Handle cases where only one class is present in the batch/epoch
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
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

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(torch.sigmoid(logits).cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def train_model(
    load_cached_data=True,
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    patience=PATIENCE,
    max_samples=None,
):
    """
    Main function to train the HRLN-Net model.
    Includes data loading, model initialization, training loop, and early stopping.
    """
    seed_everything(SEED)

    # 1. Load Data
    # load_processed_data handles caching internally
    X_train, y_train, _ = load_processed_data(
        "train", load_cached_data=load_cached_data, max_samples=max_samples
    )
    X_val, y_val, _ = load_processed_data(
        "val", load_cached_data=load_cached_data, max_samples=max_samples
    )

    train_dataset = BraTSDataset(X_train, y_train)
    val_dataset = BraTSDataset(X_val, y_val)

    # Use num_workers=0 or low number to avoid potential shared memory issues in some envs,
    # but 4 is generally fine if configured. Using 2 for safety.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 2. Setup Model, Loss, Optimizer
    model = HRLNNet().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # 3. Training Loop with Early Stopping
    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    print(f"Starting training on device: {DEVICE}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    for epoch in range(num_epochs):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE
        )
        val_loss, val_auc = validate(model, val_loader, criterion, DEVICE)

        # Print full precision metrics as requested
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss} | Train AUC: {train_auc} | "
            f"Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Checkpoint & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("  -> New best model saved!")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Val AUC: {best_auc}")
    return best_model_path
