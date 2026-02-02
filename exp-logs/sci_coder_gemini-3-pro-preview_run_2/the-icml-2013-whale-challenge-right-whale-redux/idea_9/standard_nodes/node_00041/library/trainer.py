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
    Executes the training and SWA procedure for a single fold.

    Args:
        fold_idx (int): The index of the current fold.
        train_loader (DataLoader): Loader for training data.
        val_loader (DataLoader): Loader for validation data.

    Returns:
        model (nn.Module): The trained SWA model (on CPU to save GPU memory).
        best_auc (float): The AUC score of the SWA model on the validation set.
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

    # Optimizer with low weight decay as per strategy
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 2. Initialize SWA
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=Config.SWA_LR)

    # Standard scheduler for the initial convergence phase
    # We want to anneal from LEARNING_RATE down to SWA_LR or similar before SWA starts
    # CosineAnnealingLR is a good choice.
    # T_max is set to SWA_START_EPOCH to align the schedule.
    scheduler = CosineAnnealingLR(
        optimizer, T_max=Config.SWA_START_EPOCH, eta_min=Config.SWA_LR
    )

    # 3. Training Loop
    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validation (on base model)
        val_loss, val_auc = validate_one_epoch(model, val_loader, criterion, device)

        print(
            f"Fold {fold_idx} | Epoch {epoch}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc}"
        )

        # SWA Logic
        if epoch >= Config.SWA_START_EPOCH:
            print(f"Fold {fold_idx} | Epoch {epoch} | Updating SWA model...")
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

    # 4. Finalize SWA
    print(f"Fold {fold_idx} | Finalizing SWA (updating Batch Norm statistics)...")
    # update_bn requires the train_loader to pass data through the network
    # to compute running mean/variance for BN layers.
    update_bn(train_loader, swa_model, device=device)

    # 5. Evaluate SWA Model
    print(f"Fold {fold_idx} | Validating SWA Model...")
    swa_val_loss, swa_val_auc = validate_one_epoch(
        swa_model, val_loader, criterion, device
    )

    print_metric(f"Fold {fold_idx} Final SWA AUC", swa_val_auc)

    # 6. Save Model
    save_path = os.path.join(Config.WORKING_DIR, f"swa_model_fold_{fold_idx}.pth")
    torch.save(swa_model.state_dict(), save_path)
    print(f"Fold {fold_idx} | Model saved to {save_path}")

    # Move to CPU to free up GPU memory for next fold/inference
    swa_model.cpu()

    return swa_model, swa_val_auc
