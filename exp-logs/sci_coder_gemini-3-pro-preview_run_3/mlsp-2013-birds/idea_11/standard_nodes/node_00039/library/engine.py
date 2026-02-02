import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import calculate_roc_auc, seed_everything
from library.models import get_model


def mixup_data(x, y, alpha=0.2, device=Config.DEVICE):
    """
    Applies Mixup augmentation to the batch.
    Returns mixed inputs, pairs of targets, and the mixing coefficient lambda.
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
    Computes the Mixup loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch using Mixup regularization.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        # Apply Mixup
        mixed_images, targets_a, targets_b, lam = mixup_data(
            images, labels, alpha=0.2, device=device
        )

        optimizer.zero_grad()
        outputs = model(mixed_images)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    return running_loss / dataset_size


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns loss, AUC score, and predictions.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)
            preds_list.append(probs.cpu().numpy())
            targets_list.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    if len(preds_list) > 0:
        preds = np.concatenate(preds_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)
        auc = calculate_roc_auc(targets, preds)
    else:
        preds = np.array([])
        auc = 0.0

    return epoch_loss, auc, preds


def train_fold(fold_idx, model_name, train_loader, valid_loader, device=Config.DEVICE):
    """
    Orchestrates training for a single fold and model architecture.
    Implements Early Stopping and saves the best model checkpoint.
    """
    # Reproducibility
    seed_everything(Config.SEED + fold_idx)

    # Initialize Model
    model = get_model(
        model_name, num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED
    )
    model = model.to(device)

    # Optimizer & Scheduler
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.T_MAX)

    # Criterion (BCEWithLogitsLoss is standard for multi-label)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(
        Config.CHECKPOINT_DIR, f"{model_name}_fold_{fold_idx}_best.pth"
    )

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Step scheduler (CosineAnnealingLR is typically stepped per epoch)
        scheduler.step()

        val_loss, val_auc, _ = validate(model, valid_loader, criterion, device)

        print(
            f"[{model_name} Fold {fold_idx} Epoch {epoch+1}] Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.10f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    return best_auc, best_model_path


def inference(model_name, checkpoint_path, test_loader, device=Config.DEVICE):
    """
    Performs inference on the test set using a trained model.
    Returns recording IDs and predicted probabilities.
    """
    model = get_model(model_name, num_classes=Config.NUM_CLASSES, pretrained=False)

    # Load weights
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    preds_list = []
    rec_ids_list = []

    with torch.no_grad():
        for images, _, rec_ids in test_loader:
            images = images.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs)

            preds_list.append(probs.cpu().numpy())
            rec_ids_list.append(rec_ids.numpy())

    preds = np.concatenate(preds_list, axis=0)
    rec_ids = np.concatenate(rec_ids_list, axis=0)

    return rec_ids, preds


def save_submission(rec_ids, preds, save_path):
    """
    Formats the predictions into the required CSV format and saves to disk.

    Format:
    Id,Probability
    rec_id*100 + species_id, probability
    """
    ids = []
    probs = []

    num_classes = preds.shape[1]

    for i, rid in enumerate(rec_ids):
        for species_idx in range(num_classes):
            # Construct the combined ID
            combined_id = int(rid * 100 + species_idx)

            ids.append(combined_id)
            probs.append(preds[i, species_idx])

    df = pd.DataFrame({"Id": ids, "Probability": probs})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
