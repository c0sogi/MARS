import os
import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import get_device
from library.models import get_model


def mixup_data(x, y, alpha=1.0, device="cuda"):
    """
    Applies Mixup augmentation to the batch.
    Returns mixed inputs, pairs of targets, and lambda value.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Computes the Mixup loss (linear combination of losses).
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    """
    Runs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # Apply Mixup if enabled
        if Config.USE_MIXUP and Config.MIXUP_ALPHA > 0:
            images, labels_a, labels_b, lam = mixup_data(
                images, labels, Config.MIXUP_ALPHA, device
            )
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC score.
    """
    model.eval()
    running_loss = 0.0
    preds_list = []
    targets_list = []
    dataset_size = len(loader.dataset)

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)
            preds_list.append(probs.cpu().numpy())
            targets_list.append(labels.cpu().numpy())

    total_loss = running_loss / dataset_size

    if len(preds_list) > 0:
        preds = np.concatenate(preds_list)
        targets = np.concatenate(targets_list)

        # Compute Macro ROC AUC
        # Handle potential edge cases where a class is not present in validation
        try:
            auc = roc_auc_score(targets, preds, average="macro")
        except ValueError:
            # Fallback if AUC cannot be computed (e.g., only one class present or all same label)
            auc = 0.5
    else:
        auc = 0.0

    return total_loss, auc


def train_fold(fold_idx, model_name, train_loader, val_loader, device):
    """
    Orchestrates training for a single fold.
    Handles model initialization, validation, and saving the best model.
    """
    print(f"Starting training for Fold {fold_idx} - Model: {model_name}")

    # Initialize Model
    model = get_model(model_name, pretrained=True, device=device)

    # Calculate pos_weight for BCEWithLogitsLoss to handle class imbalance
    # Access labels from the dataset (BirdDataset)
    all_labels = train_loader.dataset.labels
    pos_counts = np.sum(all_labels, axis=0)
    total_counts = len(all_labels)
    neg_counts = total_counts - pos_counts

    # Avoid division by zero and create weight tensor
    pos_counts = np.maximum(pos_counts, 1)
    pos_weight_val = neg_counts / pos_counts
    pos_weight = torch.tensor(pos_weight_val, dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    best_auc = 0.0
    save_name = f"model_fold{fold_idx}_{model_name}.pth"
    save_path = os.path.join(Config.WORK_DIR, save_name)

    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        # Validate every epoch to catch the best model
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Save Best Model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), save_path)

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(
                f"Fold {fold_idx} Epoch {epoch+1}/{Config.NUM_EPOCHS}: Train Loss {train_loss:.4f} Val Loss {val_loss:.4f} Val AUC {val_auc:.4f} (Best: {best_auc:.4f})"
            )

    print(f"Fold {fold_idx} Finished. Best Val AUC: {best_auc:.4f}")
    return best_auc
