import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
import numpy as np

from library.config import Config
from library.utils import (
    seed_everything,
    mixup_data,
    mixup_criterion,
    calculate_roc_auc,
    update_swa_bn,
    save_checkpoint,
)
from library.models import CactusRepVGG, CactusResNet, CactusMicroNeXt


def get_model(model_name):
    """
    Factory function to instantiate models based on name.
    """
    if model_name == "CactusRepVGG":
        return CactusRepVGG(num_classes=Config.NUM_CLASSES)
    elif model_name == "CactusResNet":
        return CactusResNet(num_classes=Config.NUM_CLASSES)
    elif model_name == "CactusMicroNeXt":
        return CactusMicroNeXt(num_classes=Config.NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    """
    Trains the model for one epoch using Mixup augmentation.
    """
    model.train()
    running_loss = 0.0

    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        # Apply Mixup
        mixed_images, targets_a, targets_b, lam = mixup_data(
            images, labels, alpha=Config.MIXUP_ALPHA, device=device
        )

        optimizer.zero_grad()
        outputs = mixed_images  # Variable name reuse for clarity in flow
        outputs = model(mixed_images)

        # Prepare targets for BCEWithLogitsLoss (needs float, shape [N, 1])
        targets_a = targets_a.view(-1, 1)
        targets_b = targets_b.view(-1, 1)

        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss, ROC AUC, predictions, and targets.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).view(-1, 1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            # Apply sigmoid to get probabilities
            preds = torch.sigmoid(outputs)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    avg_loss = running_loss / len(loader)
    auc = calculate_roc_auc(all_targets, all_preds)

    return avg_loss, auc, all_preds, all_targets


def run_fold(fold, model_name, train_loader, val_loader):
    """
    Orchestrates the training process for a single fold.
    Handles Optimizer, Scheduler, SWA, and Checkpointing.
    """
    print(f"Starting Run: Fold {fold} | Model {model_name}")

    device = torch.device(Config.DEVICE)
    seed_everything(Config.SEED + fold)

    # Initialize Model
    model = get_model(model_name)
    model = model.to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Loss Function
    criterion = nn.BCEWithLogitsLoss()

    # Standard Scheduler (Cosine Annealing)
    # Runs until SWA starts
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.SWA_START_EPOCH, eta_min=1e-6
    )

    # SWA Setup
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(
        optimizer,
        swa_lr=Config.LEARNING_RATE * 0.5,
        anneal_epochs=3,
        anneal_strategy="cos",
    )

    # Tracking
    best_auc = 0.0
    best_model_path = Config.get_checkpoint_path(f"{model_name}_best", fold)
    swa_model_path = Config.get_checkpoint_path(f"{model_name}_swa", fold)

    best_val_preds = None
    best_val_targets = None

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        # Scheduler Logic
        if epoch >= Config.SWA_START_EPOCH:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            mode = "SWA"
        else:
            scheduler.step()
            mode = "Standard"

        # Validate (using the current standard model weights)
        val_loss, val_auc, val_preds, val_targets = validate(
            model, val_loader, criterion, device
        )

        # Checkpoint Best Model
        if val_auc > best_auc:
            best_auc = val_auc
            best_val_preds = val_preds
            best_val_targets = val_targets
            save_checkpoint(model, best_model_path)
            print(
                f"Epoch {epoch+1} [{mode}] | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val AUC: {val_auc} *Best*"
            )
        else:
            print(
                f"Epoch {epoch+1} [{mode}] | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val AUC: {val_auc}"
            )

    # Finalize SWA
    print("Finalizing SWA...")
    update_swa_bn(train_loader, swa_model, device)

    # Validate SWA Model
    swa_loss, swa_auc, swa_preds, swa_targets = validate(
        swa_model, val_loader, criterion, device
    )
    print(f"SWA Results | Val Loss: {swa_loss:.5f} | Val AUC: {swa_auc}")

    save_checkpoint(swa_model, swa_model_path)

    return {
        "best_auc": best_auc,
        "swa_auc": swa_auc,
        "best_preds": best_val_preds,
        "swa_preds": swa_preds,
        "targets": best_val_targets,
    }
