import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.utils import (
    set_seed,
    ModelEMA,
    Mixup,
    MetricMonitor,
    save_checkpoint,
)
from library.dataset import (
    load_data,
    PathologyDataset,
    get_transforms,
)
from library.models import get_model


def train_one_epoch(model, loader, optimizer, criterion, device, ema_model=None):
    """
    Trains the model for one epoch using Mixup and EMA.
    """
    model.train()
    monitor = MetricMonitor()
    mixup_fn = Mixup(alpha=Config.MIXUP_ALPHA)

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Apply Mixup
        mixed_inputs, targets_a, targets_b, lam = mixup_fn(inputs, targets)

        optimizer.zero_grad()
        outputs = model(mixed_inputs)

        # Prepare targets for BCEWithLogitsLoss (needs float and shape matching)
        targets_a = targets_a.view(-1, 1).float()
        targets_b = targets_b.view(-1, 1).float()

        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(
            outputs, targets_b
        )

        loss.backward()
        optimizer.step()

        # Update EMA model
        if ema_model:
            ema_model.update(model)

        monitor.update(loss.item(), inputs.size(0))

    return monitor.get_avg_loss()


def validate(model, loader, criterion, device):
    """
    Validates the model on the validation set.
    """
    model.eval()
    monitor = MetricMonitor()

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            # Calculate Loss
            targets_float = targets.view(-1, 1).float()
            loss = criterion(outputs, targets_float)

            # Calculate AUC (needs sigmoid probabilities)
            preds = torch.sigmoid(outputs)

            monitor.update(loss.item(), inputs.size(0))
            monitor.update_predictions(preds, targets)

    return monitor.get_avg_loss(), monitor.get_auc()


def run_training():
    """
    Orchestrates the training of the Heterogeneous Deep Ensemble.
    Trains multiple architectures using Stratified K-Fold Cross-Validation.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Starting training on device: {device}")

    # 1. Prepare Data for Cross-Validation
    # We use only the training set for CV to avoid leakage into the hold-out validation set
    print("Loading training dataset for K-Fold CV...")
    train_images, train_labels = load_data("train")

    print(f"Total training samples: {len(train_labels)}")

    # 2. Initialize K-Fold
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # 3. Iterate over Architectures
    for arch_name in Config.MODEL_ARCHS:
        print(f"\n{'='*40}")
        print(f"Training Architecture: {arch_name}")
        print(f"{'='*40}")

        # 4. Iterate over Folds
        for fold, (train_idx, val_idx) in enumerate(
            skf.split(train_images, train_labels)
        ):
            print(f"\n--- Fold {fold + 1}/{Config.N_FOLDS} ---")

            # Split Data
            X_train, y_train = train_images[train_idx], train_labels[train_idx]
            X_val, y_val = train_images[val_idx], train_labels[val_idx]

            # Create Datasets
            train_dataset = PathologyDataset(
                X_train, y_train, transforms=get_transforms("train")
            )
            val_dataset = PathologyDataset(
                X_val, y_val, transforms=get_transforms("val")
            )

            # Create DataLoaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=True,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
                drop_last=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
                drop_last=False,
            )

            # Initialize Model
            model = get_model(arch_name, pretrained=True, num_classes=1)
            model = model.to(device)

            # Initialize EMA
            ema_model = None
            if Config.USE_EMA:
                ema_model = ModelEMA(model, decay=Config.EMA_DECAY, device=device)

            # Optimizer & Scheduler
            optimizer = optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=Config.EPOCHS
            )
            criterion = nn.BCEWithLogitsLoss()

            # Training Loop
            best_auc = 0.0

            for epoch in range(Config.EPOCHS):
                train_loss = train_one_epoch(
                    model, train_loader, optimizer, criterion, device, ema_model
                )

                # Validate using EMA model if available, otherwise standard model
                eval_model = ema_model.module if ema_model else model
                val_loss, val_auc = validate(eval_model, val_loader, criterion, device)

                scheduler.step()

                print(
                    f"Epoch {epoch + 1}/{Config.EPOCHS} | "
                    f"Train Loss: {train_loss:.5f} | "
                    f"Val Loss: {val_loss:.5f} | "
                    f"Val AUC: {val_auc:.8f}"
                )

                # Save Checkpoint
                is_best = val_auc > best_auc
                if is_best:
                    best_auc = val_auc

                # Construct filename: checkpoint_{arch}_fold_{fold}.pth
                # save_checkpoint will create best_model_{arch}_fold_{fold}.pth if is_best=True
                filename = f"checkpoint_{arch_name}_fold_{fold}.pth"

                # Prepare state dict
                state = {
                    "epoch": epoch + 1,
                    "arch": arch_name,
                    "fold": fold,
                    "state_dict": model.state_dict(),
                    "best_auc": best_auc,
                    "optimizer": optimizer.state_dict(),
                }
                if ema_model:
                    state["ema_state_dict"] = ema_model.module.state_dict()

                save_checkpoint(
                    state,
                    is_best,
                    Config.CHECKPOINT_DIR,
                    filename=filename,
                )

            print(f"Fold {fold + 1} Best AUC: {best_auc:.8f}")

            # Cleanup to save memory
            del model, ema_model, optimizer, scheduler, train_loader, val_loader
            torch.cuda.empty_cache()

    print("\nTraining Complete.")
