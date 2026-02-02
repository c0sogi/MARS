import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from library.config import Config
from library.dataset import BirdDataset, get_transforms, load_dataframe
from library.model import BirdClassifier
from library.utils import set_seed, calculate_roc_auc, save_checkpoint


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)

            # Apply sigmoid to convert logits to probabilities
            probs = torch.sigmoid(logits)

            all_targets.append(labels.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    val_loss = running_loss / len(loader.dataset)

    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    auc_score = calculate_roc_auc(all_targets, all_preds)

    return val_loss, auc_score


def run_training(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    debug=Config.DEBUG,
    save_path=Config.MODEL_SAVE_PATH,
):
    """
    Main training function.

    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for data loaders.
        debug (bool): Whether to run in debug mode (subset of data).
        save_path (str): Path to save the best model checkpoint.
    """
    set_seed(Config.SEED)
    device = Config.DEVICE

    print(f"Starting training on device: {device}")

    # --- Data Loading ---
    df_train = load_dataframe(Config.TRAIN_CSV, debug=debug)
    df_val = load_dataframe(Config.VAL_CSV, debug=debug)

    train_dataset = BirdDataset(df_train, transforms=get_transforms("train"))
    val_dataset = BirdDataset(df_val, transforms=get_transforms("val"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- Model Setup ---
    model = BirdClassifier(
        model_name=Config.MODEL_NAME,
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES,
        dropout_rate=Config.DROPOUT_RATE,
    )
    model.to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # --- Loss Function Configuration ---
    # Calculate positive weights if enabled to handle class imbalance
    pos_weight = None
    if Config.USE_POS_WEIGHT:
        label_cols = [c for c in df_train.columns if c.startswith("species_")]
        labels = df_train[label_cols].values

        # Calculate pos_weight: number of negatives / number of positives
        # Add epsilon to avoid division by zero
        pos_counts = np.sum(labels, axis=0)
        neg_counts = len(labels) - pos_counts
        weights = neg_counts / (pos_counts + 1e-6)

        pos_weight = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- Training Loop ---
    best_score = -1.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_score = validate(model, val_loader, criterion, device)

        # Step the scheduler
        scheduler.step()

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val AUC: {val_score}"
        )

        # Save Checkpoint
        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_score, filename=save_path
            )
            print(f"New best model saved with AUC: {best_score}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    print(f"Training complete. Best Validation AUC: {best_score}")
