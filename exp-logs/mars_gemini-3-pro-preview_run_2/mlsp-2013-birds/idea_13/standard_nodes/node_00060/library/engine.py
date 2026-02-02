import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from library.config import Config
from library.utils import (
    seed_everything,
    calculate_roc_auc,
    mixup_data,
    get_pos_weights,
)
from library.data import get_fold_dataloaders
from library.models import BirdClassifier


def train_one_epoch(model, optimizer, data_loader, criterion, device):
    """
    Trains the model for one epoch using Mixup augmentation.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        # Apply Mixup
        images, targets_a, targets_b, lam = mixup_data(
            images, labels, alpha=Config.MIXUP_ALPHA, device=device
        )

        optimizer.zero_grad()

        outputs = model(images)

        # Mixup Loss
        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(
            outputs, targets_b
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, data_loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    roc_auc = calculate_roc_auc(all_targets, all_preds)

    return epoch_loss, roc_auc


def run_fold(fold_idx, model_name):
    """
    Runs the training and validation loop for a specific fold and model architecture.

    Args:
        fold_idx (int): The fold index (0-4).
        model_name (str): The model architecture name (e.g., 'resnet18').
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Starting training for {model_name} - Fold {fold_idx}")

    # 1. Data Loading
    train_loader, val_loader = get_fold_dataloaders(fold_idx, model_name)

    # 2. Model Initialization
    model = BirdClassifier(
        model_name=model_name, num_classes=Config.NUM_SPECIES, pretrained=True
    )
    model.to(device)

    # 3. Loss Function with Positive Weights
    # Extract labels from dataset to compute weights
    y_train = train_loader.dataset.labels
    pos_weights = get_pos_weights(y_train, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # 4. Optimizer
    # AdamW with constant learning rate as per strategy
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 5. Training Loop with Early Stopping
    best_roc_auc = 0.0
    patience = 15
    patience_counter = 0

    model_save_path = os.path.join(
        Config.WORKING_DIR, f"model_{model_name}_fold_{fold_idx}.pth"
    )

    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, optimizer, train_loader, criterion, device)
        val_loss, val_roc_auc = validate(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val ROC AUC: {val_roc_auc}"
        )

        # Checkpoint and Early Stopping
        if val_roc_auc > best_roc_auc:
            best_roc_auc = val_roc_auc
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved to {model_save_path}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Fold {fold_idx} finished. Best Val ROC AUC: {best_roc_auc}")

    # Clean up to save memory
    del model, optimizer, train_loader, val_loader
    torch.cuda.empty_cache()

    return best_roc_auc
