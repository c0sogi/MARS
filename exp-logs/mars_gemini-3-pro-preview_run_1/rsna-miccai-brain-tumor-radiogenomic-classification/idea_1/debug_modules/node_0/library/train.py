import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np

from library.config import Config
from library.utils import seed_everything
from library.dataset import MGMTDataset, get_transforms
from library.model import MGMTClassifier


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    for batch_idx, (images, targets, _) in enumerate(loader):
        images = images.to(device)
        # Targets shape: (B,) -> (B, 1) to match logits
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        # Accumulate batch loss (multiply by batch size)
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
    dataset_size = len(loader.dataset)

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, targets)

            running_loss += loss.item() * images.size(0)

            # Apply sigmoid to get probabilities for AUC
            probs = torch.sigmoid(logits)

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(probs.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    # Calculate AUC
    # Handle potential edge case if batch only has one class (though unlikely with full val set)
    try:
        auc_score = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc_score = 0.5

    return epoch_loss, auc_score


def run_training(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    patience=Config.PATIENCE,
    device=Config.DEVICE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
):
    """
    Main training loop with early stopping.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    Config.setup()  # Ensure output directories exist

    print(f"Initializing training on device: {device}")

    # 2. Data Loading
    # The MGMTDataset handles caching logic internally based on load_cached_data flag
    train_dataset = MGMTDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        split="train",
        transform=get_transforms("train"),
        load_cached_data=load_cached_data,
    )

    val_dataset = MGMTDataset(
        metadata_path=Config.VAL_METADATA_PATH,
        split="val",
        transform=get_transforms("val"),
        load_cached_data=load_cached_data,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 3. Model Setup
    model = MGMTClassifier(
        model_name=Config.BACKBONE,
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES,
        dropout_rate=Config.DROPOUT_RATE,
    )
    model.to(device)

    # 4. Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # 5. Training Loop
    best_auc = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        print(f"--- Epoch {epoch + 1}/{epochs} ---")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val AUC: {val_auc}")

        # Early Stopping Logic
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"New best model saved to {Config.MODEL_PATH}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training finished. Best Validation AUC: {best_auc}")
    return best_auc
