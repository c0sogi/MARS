import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from library.config import Config
from library.utils import seed_everything, calculate_roc_auc
from library.data import get_dataloaders
from library.model import MultiViewResNet


def train_one_epoch(
    model, loader, optimizer, criterion, device, use_mixup, mixup_alpha
):
    """
    Trains the model for one epoch using Mixup regularization if enabled.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        # Apply Mixup
        if use_mixup and mixup_alpha > 0:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            index = torch.randperm(batch_size).to(device)

            mixed_images = lam * images + (1 - lam) * images[index]
            labels_a, labels_b = labels, labels[index]

            outputs = model(mixed_images)
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(
                outputs, labels_b
            )
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to logits to get probabilities for AUC calculation
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        epoch_auc = calculate_roc_auc(all_targets, all_preds)
    else:
        epoch_auc = 0.0

    return epoch_loss, epoch_auc


def run_fold(fold_idx, debug=False):
    """
    Runs the training and validation loop for a specific fold.
    """
    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)

    # Initialize Model
    model = MultiViewResNet()
    model = model.to(device)

    # Optimizer and Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    criterion = nn.BCEWithLogitsLoss()

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.SCHEDULER_T_MAX, eta_min=Config.SCHEDULER_ETA_MIN
    )

    # DataLoaders
    # We only need train and val loaders for the training loop
    train_loader, val_loader, _ = get_dataloaders(
        fold_idx, load_cached_data=True, debug=debug
    )

    # Early Stopping Variables
    best_val_auc = -1.0
    patience_counter = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, f"fold_{fold_idx}_best.pth")

    print(f"Starting training for Fold {fold_idx}...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            Config.USE_MIXUP,
            Config.MIXUP_ALPHA,
        )

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        # Print Metrics (Full Precision)
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} - "
            f"Train Loss: {train_loss}, "
            f"Val Loss: {val_loss}, "
            f"Val AUC: {val_auc}"
        )

        # Early Stopping & Checkpointing
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Fold {fold_idx} finished. Best Val AUC: {best_val_auc}")

    # Clear memory
    del model, optimizer, scheduler, train_loader, val_loader
    torch.cuda.empty_cache()
