import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import seed_everything, mixup_data, mixup_criterion
from library.dataset import get_dataloaders
from library.model import MILResNet18


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        batch_size = inputs.size(0)

        optimizer.zero_grad()

        if Config.USE_MIXUP:
            mixed_inputs, targets_a, targets_b, lam = mixup_data(
                inputs, targets, Config.MIXUP_ALPHA, device
            )
            outputs = model(mixed_inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns the average loss and macro-averaged ROC AUC score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            batch_size = inputs.size(0)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to convert logits to probabilities
            probs = torch.sigmoid(outputs)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    val_loss = running_loss / dataset_size

    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    # Compute ROC AUC
    # Calculate AUC per class and average only valid classes
    aucs = []
    n_classes = all_targets.shape[1]
    for i in range(n_classes):
        # Only calculate AUC if the class has at least one positive and one negative sample
        if len(np.unique(all_targets[:, i])) > 1:
            try:
                score = roc_auc_score(all_targets[:, i], all_preds[:, i])
                aucs.append(score)
            except ValueError:
                pass

    if len(aucs) > 0:
        val_auc = np.mean(aucs)
    else:
        val_auc = 0.5
        print(
            "Warning: ROC AUC calculation failed (no valid classes). Defaulting to 0.5."
        )

    return val_loss, float(val_auc)


def run_fold(fold, load_cached_data=True):
    """
    Runs the training and validation loop for a specific fold.
    """
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Running Fold {fold} on {device}")

    # Get DataLoaders
    train_loader, val_loader, _ = get_dataloaders(
        fold=fold, load_cached_data=load_cached_data
    )

    # Initialize Model
    model = MILResNet18()
    model.to(device)

    # Optimizer and Scheduler
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR)

    # Loss Function
    # BCEWithLogitsLoss combines Sigmoid layer and BCELoss in one single class
    criterion = nn.BCEWithLogitsLoss()

    # Training Loop Variables
    best_auc = -1.0
    patience_counter = 0
    best_model_path = os.path.join(Config.CHECKPOINTS_DIR, f"fold_{fold}_best.pth")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Step the scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | LR: {current_lr:.8f} | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc}"
        )

        # Checkpointing and Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved for fold {fold} with AUC: {best_auc}")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Fold {fold} finished. Best AUC: {best_auc}")
    return best_auc
