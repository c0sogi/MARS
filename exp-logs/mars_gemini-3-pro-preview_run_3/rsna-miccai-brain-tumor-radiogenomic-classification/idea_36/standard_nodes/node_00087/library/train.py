import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from library.utils import seed_everything, get_device
from library.data import get_dataloaders
from library.model import Stacked25DNet


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for x, y in loader:
        # Move data to device
        x = x.to(device)
        y = y.to(device).float().view(-1, 1)

        # Forward pass
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * y.size(0)

        # Store predictions for AUC
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.append(probs)
        all_targets.append(y.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        # Handle case where batch has only one class
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).float().view(-1, 1)

            logits = model(x)
            loss = criterion(logits, y)

            running_loss += loss.item() * y.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(y.cpu().numpy())

    val_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    try:
        val_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        val_auc = 0.5

    return val_loss, val_auc


def run_training(
    epochs=20,
    batch_size=16,
    learning_rate=1e-4,
    patience=5,
    save_dir="./working/idea_opt",
    load_cached_data=True,
):
    """
    Main function to run the training pipeline.
    """
    # 1. Setup
    seed_everything(42)
    device = get_device()
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.pth")

    # 2. Data Loading
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=batch_size, load_cached_data=load_cached_data
    )

    # 3. Model Initialization
    model = Stacked25DNet(model_name="efficientnet_b0", pretrained=True)
    model.to(device)

    # 4. Optimizer & Loss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 5. Training Loop
    best_auc = 0.0
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(1, epochs + 1):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch}/{epochs} - "
            f"Train Loss: {train_loss}, Train AUC: {train_auc} - "
            f"Val Loss: {val_loss}, Val AUC: {val_auc}"
        )

        # Checkpointing & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with AUC: {best_auc}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping triggered after {patience} epochs without improvement."
                )
                break

    print(f"Training complete. Best Validation AUC: {best_auc}")
    return best_auc
