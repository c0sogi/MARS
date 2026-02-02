import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from library.utils import seed_everything, get_device
from library.data_loader import get_dataloaders
from library.model import BraTSEfficientNet

# ==========================================
# Constants
# ==========================================
WORKING_DIR = "./working/idea_11"
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()

        # Forward pass
        logits = model(inputs)
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        count += inputs.size(0)

    epoch_loss = running_loss / count if count > 0 else 0.0
    return epoch_loss


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
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            logits = model(inputs)
            loss = criterion(logits, targets)

            running_loss += loss.item() * inputs.size(0)
            count += inputs.size(0)

            # Apply sigmoid to get probabilities for AUC calculation
            probs = torch.sigmoid(logits)

            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    avg_loss = running_loss / count if count > 0 else 0.0

    # Concatenate all batches
    if len(all_targets) > 0:
        all_targets = np.concatenate(all_targets)
        all_probs = np.concatenate(all_probs)

        # Calculate AUC
        # Handle edge case where only one class is present in the batch
        if len(np.unique(all_targets)) > 1:
            auc_score = roc_auc_score(all_targets, all_probs)
        else:
            auc_score = 0.5
    else:
        auc_score = 0.5

    return avg_loss, auc_score


def run_training(
    epochs=20,
    batch_size=8,
    lr=1e-4,
    patience=5,
    load_cached_data=True,
    debug_limit=None,
):
    """
    Main function to orchestrate the training process.
    """
    # 1. Setup
    seed_everything(42)
    device = get_device()
    os.makedirs(WORKING_DIR, exist_ok=True)

    print(f"Device: {device}")
    print(
        f"Training Configuration: Epochs={epochs}, Batch Size={batch_size}, LR={lr}, Patience={patience}"
    )

    # 2. Data Loading
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=batch_size,
        load_cached_data=load_cached_data,
        debug_limit=debug_limit,
    )

    # 3. Model Initialization
    model = BraTSEfficientNet()
    model.to(device)

    # 4. Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 5. Training Loop
    best_auc = 0.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Print metrics (Full precision as requested)
        print(f"Epoch {epoch}/{epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val AUC: {val_auc}")

        # Checkpointing & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            print(f"New best AUC! Saving model to {MODEL_SAVE_PATH}")
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation AUC: {best_auc}")
    return best_auc
