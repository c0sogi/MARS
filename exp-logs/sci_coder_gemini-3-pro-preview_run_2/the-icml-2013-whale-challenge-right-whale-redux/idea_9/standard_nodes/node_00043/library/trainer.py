import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
import numpy as np
from tqdm import (
    tqdm,
)  # Not strictly required by prompt but useful, will suppress if needed or just not use to be safe with "silent" reqs

from library.config import Config
from library.utils import set_seed, calculate_roc_auc, print_metric
from library.model_factory import WhaleEfficientNet


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(1)  # Ensure (B, 1) shape for BCE

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        count += inputs.size(0)

    return running_loss / count


def validate_one_epoch(model, loader, criterion, device):
    """
    Validates the model on the validation set.
    Returns average loss and ROC AUC score.
    """
    model.eval()
    running_loss = 0.0
    count = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            count += inputs.size(0)

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(outputs)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    avg_loss = running_loss / count

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    auc_score = calculate_roc_auc(all_targets, all_preds)

    return avg_loss, auc_score


def run_fold(fold_idx, train_loader, val_loader):
    """
    Executes the training with Early Stopping for a single fold.

    Args:
        fold_idx (int): The index of the current fold.
        train_loader (DataLoader): Loader for training data.
        val_loader (DataLoader): Loader for validation data.

    Returns:
        model (nn.Module): The best trained model (on CPU to save GPU memory).
        best_auc (float): The best AUC score on the validation set.
    """
    print(f"\n=== Starting Fold {fold_idx} ===")
    set_seed(Config.SEED + fold_idx)
    device = torch.device(Config.DEVICE)

    # 1. Initialize Model, Optimizer, Loss
    model = WhaleEfficientNet(
        model_name=Config.MODEL_NAME, pretrained=Config.PRETRAINED
    )
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    # Optimizer with low weight decay
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: Cosine Annealing
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)

    # Early Stopping Variables
    best_auc = 0.0
    best_model_state = None
    patience_counter = 0

    # 3. Training Loop
    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validation
        val_loss, val_auc = validate_one_epoch(model, val_loader, criterion, device)

        print(
            f"Fold {fold_idx} | Epoch {epoch}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc}"
        )

        scheduler.step()

        # Early Stopping Logic
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict()
            patience_counter = 0
            print(f"Fold {fold_idx} | New Best AUC: {best_auc}")
        else:
            patience_counter += 1
            print(f"Fold {fold_idx} | Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print(f"Fold {fold_idx} | Early stopping triggered at epoch {epoch}")
            break

    # 4. Load Best Model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print_metric(f"Fold {fold_idx} Best AUC", best_auc)

    # 5. Save Model
    save_path = os.path.join(Config.WORKING_DIR, f"best_model_fold_{fold_idx}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Fold {fold_idx} | Model saved to {save_path}")

    # Move to CPU to free up GPU memory for next fold/inference
    model.cpu()

    return model, best_auc
