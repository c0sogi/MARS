import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library import utils, dataset, model as model_lib


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs):
    """
    Trains the model for one epoch.
    Updates the DropBlock probability based on a linear schedule.
    """
    model.train()

    # Linear schedule for DropBlock probability: 0 -> Config.DROPBLOCK_PROB_MAX
    # We use (total_epochs - 1) to ensure we reach max prob at the last epoch,
    # or we can cap it. Here we scale by progress.
    if total_epochs > 1:
        progress = epoch / (total_epochs - 1)
    else:
        progress = 1.0

    current_drop_prob = progress * Config.DROPBLOCK_PROB_MAX
    # Clamp to ensure it doesn't exceed max config
    current_drop_prob = min(Config.DROPBLOCK_PROB_MAX, max(0.0, current_drop_prob))

    # Update the model's DropBlock layers
    model.set_dropblock_prob(current_drop_prob)

    losses = utils.AverageMeter()

    for inputs, angles, targets in loader:
        inputs = inputs.to(device)
        angles = angles.to(device)
        targets = targets.to(device).unsqueeze(1)  # Shape (N, 1) for BCE

        optimizer.zero_grad()

        outputs = model(inputs, angles)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inputs.size(0))

    return losses.avg


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns Log Loss and Accuracy.
    """
    model.eval()
    losses = utils.AverageMeter()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, angles, targets in loader:
            inputs = inputs.to(device)
            angles = angles.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(inputs, angles)
            loss = criterion(outputs, targets)

            losses.update(loss.item(), inputs.size(0))

            # Apply sigmoid to get probabilities for metrics
            preds = torch.sigmoid(outputs)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()

    # Calculate metrics
    log_loss_val = utils.calculate_log_loss(all_targets, all_preds)
    accuracy = utils.calculate_accuracy(all_targets, all_preds)

    return log_loss_val, accuracy


def train_model():
    """
    Main training routine.
    Performs 5-Fold Cross-Validation, training the Dual-Polarity DropBlock SE-CNN.
    """
    utils.set_seed(Config.SEED)

    print("Loading and preparing data...")
    # Load cached data or process from scratch
    data = dataset.prepare_data(load_cached_data=True)

    # Recombine train and val splits to perform Stratified K-Fold manually
    # The provided metadata splits are 80/20, but we want to do 5-fold CV on the whole labeled set.
    X_full = np.concatenate([data["train"]["X"], data["val"]["X"]], axis=0)
    y_full = np.concatenate([data["train"]["y"], data["val"]["y"]], axis=0)
    angle_full = np.concatenate([data["train"]["angle"], data["val"]["angle"]], axis=0)

    if Config.DEBUG:
        print(f"Debug mode: Reducing dataset size to {Config.DEBUG_SIZE}")
        X_full = X_full[: Config.DEBUG_SIZE]
        y_full = y_full[: Config.DEBUG_SIZE]
        angle_full = angle_full[: Config.DEBUG_SIZE]

    # Initialize Cross-Validation
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    fold_best_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        print(f"\n{'='*20} Fold {fold} {'='*20}")

        # Split data for this fold
        X_train, X_val = X_full[train_idx], X_full[val_idx]
        y_train, y_val = y_full[train_idx], y_full[val_idx]
        angle_train, angle_val = angle_full[train_idx], angle_full[val_idx]

        # Create Datasets
        train_ds = dataset.IcebergDataset(
            X_train,
            angle_train,
            labels=y_train,
            transform=dataset.get_transforms("train"),
        )
        val_ds = dataset.IcebergDataset(
            X_val, angle_val, labels=y_val, transform=dataset.get_transforms("test")
        )

        # Create DataLoaders
        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model
        model = model_lib.DualPolarityDropBlockSECNN().to(Config.DEVICE)

        # Optimizer (AdamW with constant LR)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Loss Function
        criterion = nn.BCEWithLogitsLoss()

        # Training Loop Variables
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(Config.NUM_EPOCHS):
            start_time = time.time()

            # Train
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                Config.DEVICE,
                epoch,
                Config.NUM_EPOCHS,
            )

            # Validate
            val_loss, val_acc = validate(model, val_loader, criterion, Config.DEVICE)

            elapsed = time.time() - start_time

            # Print metrics (Full precision for validation loss)
            print(
                f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
                f"Time: {elapsed:.1f}s | "
                f"Train Loss: {train_loss:.5f} | "
                f"Val Loss: {val_loss} | "
                f"Val Acc: {val_acc:.5f}"
            )

            # Checkpoint and Early Stopping
            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            utils.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_loss": best_loss,
                    "fold": fold,
                },
                is_best,
                fold,
            )

            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        print(f"Fold {fold} Best Log Loss: {best_loss}")
        fold_best_scores.append(best_loss)

    print("\n" + "=" * 40)
    print(f"CV Average Log Loss: {np.mean(fold_best_scores)}")
    print("=" * 40)
