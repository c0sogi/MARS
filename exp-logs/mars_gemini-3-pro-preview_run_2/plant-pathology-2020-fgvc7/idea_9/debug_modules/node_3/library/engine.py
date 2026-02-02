import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

from library.config import Config
from library.utils import seed_everything, calculate_roc_auc, get_class_weights
from library.dataset import get_dataloaders
from library.model import AppleDiseaseModel


def train_one_epoch(model, optimizer, dataloader, device, criterion):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, targets, _ in dataloader:
        images = images.to(device)
        targets = targets.to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def valid_one_epoch(model, dataloader, device, criterion):
    """
    Validates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets, _ in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to logits for metric calculation
            preds = torch.sigmoid(outputs)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    epoch_auc = calculate_roc_auc(all_targets, all_preds)

    return epoch_loss, epoch_auc, all_preds, all_targets


def run_fold(fold, model_config):
    """
    Runs the training and validation loop for a specific fold.

    Args:
        fold (int): The fold index.
        model_config (dict): Configuration dictionary for the model architecture.
    """
    seed_everything(Config.seed)

    device = Config.device
    print(f"--- Starting Fold {fold} | Model: {model_config['name']} ---")

    # 1. Data Loaders
    train_loader, val_loader, _ = get_dataloaders(
        fold=fold, image_size=model_config["image_size"], batch_size=Config.batch_size
    )

    # 2. Model Setup
    model = AppleDiseaseModel(
        model_name=model_config["name"],
        pretrained=True,
        num_classes=Config.num_classes,
        drop_rate=model_config["dropout_rate"],
        drop_path_rate=model_config["drop_path_rate"],
        use_gem=model_config["use_gem"],
    )
    model.to(device)

    # 3. Loss Function with Class Weights
    # Calculate weights based on training data for this fold
    train_df = train_loader.dataset.df
    pos_weights = get_class_weights(train_df, Config.target_columns)
    pos_weights = pos_weights.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    # Standard scheduler for initial phase
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=Config.scheduler_factor,
        patience=Config.scheduler_patience,
    )

    # 5. SWA Setup
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=Config.swa_lr)

    # 6. Training Loop
    best_auc = 0.0
    best_loss = float("inf")
    early_stopping_counter = 0

    # Paths for saving models
    best_model_path = os.path.join(
        Config.working_dir, f"best_model_{model_config['name']}_fold_{fold}.pth"
    )
    swa_model_path = os.path.join(
        Config.working_dir, f"swa_model_{model_config['name']}_fold_{fold}.pth"
    )

    for epoch in range(Config.epochs):
        # Determine if we are in SWA phase
        in_swa_phase = Config.use_swa and (epoch >= Config.swa_start_epoch)

        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, device, criterion)

        # SWA Update
        if in_swa_phase:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        # Validation
        val_loss, val_auc, _, _ = valid_one_epoch(model, val_loader, device, criterion)

        # Scheduler Step (Standard phase only)
        if not in_swa_phase:
            scheduler.step(val_auc)

        # Logging
        print(
            f"Epoch {epoch+1}/{Config.epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc:.6f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
            f"SWA: {in_swa_phase}"
        )

        # Checkpointing & Early Stopping (Standard phase only)
        # In SWA phase, we continue training to collect averages
        if not in_swa_phase:
            if val_auc > best_auc:
                best_auc = val_auc
                best_loss = val_loss
                early_stopping_counter = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"  New best model saved! AUC: {best_auc:.6f}")
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= Config.early_stopping_patience:
                print("  Early stopping triggered.")
                # If SWA is enabled but we triggered early stopping before SWA start,
                # we should probably break. If SWA is coming up, we might want to continue?
                # Strategy: If early stopping hits before SWA start, we stop.
                # If we are close to SWA, maybe we should have continued, but strict ES says stop.
                if epoch < Config.swa_start_epoch:
                    break

    # 7. Finalize SWA
    if Config.use_swa:
        print("Finalizing SWA model (updating BatchNorm)...")
        # Update BN statistics for the SWA model
        update_bn(train_loader, swa_model, device=device)

        # Save SWA model
        torch.save(swa_model.state_dict(), swa_model_path)
        print(f"SWA model saved to {swa_model_path}")

    print(f"Fold {fold} finished. Best Standard AUC: {best_auc:.6f}")
