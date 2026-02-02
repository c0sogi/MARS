import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Import from provided library files
from library.config import Config
from library.utils import seed_everything
from library.dataset import BraTSDataset, get_transforms
from library.model import AAWIISNet


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Training loop for one epoch.
    """
    model.train()
    running_loss = 0.0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(1)  # Match shape (B, 1)

        optimizer.zero_grad()

        # Forward pass (outputs are logits)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate_epoch(model, loader, criterion, device):
    """
    Validation loop for one epoch. Computes Loss and ROC AUC.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

            # Apply sigmoid to get probabilities for AUC calculation
            probs = torch.sigmoid(outputs)

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(probs.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    # Calculate ROC AUC
    # Handle edge case where batch might only have one class
    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def run_training(
    train_metadata_path=Config.TRAIN_METADATA_PATH,
    val_metadata_path=Config.VAL_METADATA_PATH,
    output_dir=Config.WORKING_DIR,
    num_epochs=Config.NUM_EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    patience=5,
):
    """
    Main function to run the training pipeline with Early Stopping.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    os.makedirs(output_dir, exist_ok=True)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Load Metadata
    print(f"Loading metadata from {train_metadata_path} and {val_metadata_path}...")
    df_train = pd.read_csv(train_metadata_path)
    df_val = pd.read_csv(val_metadata_path)

    # 3. Initialize Datasets and Dataloaders
    # Note: Dataset handles caching internally via prepare_roi_cache
    train_dataset = BraTSDataset(
        df_train,
        transform=get_transforms("train"),
        is_train=True,
        cache_name="train_roi",
    )
    val_dataset = BraTSDataset(
        df_val, transform=get_transforms("val"), is_train=False, cache_name="val_roi"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batch to stabilize BatchNorm
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Training samples (slabs): {len(train_dataset)}")
    print(f"Validation samples (slabs): {len(val_dataset)}")

    # 4. Initialize Model, Loss, Optimizer
    model = AAWIISNet(pretrained=Config.PRETRAINED)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # 5. Training Loop
    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(output_dir, "best_model.pth")

    print("\nStarting training...")
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_auc = validate_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {train_loss:.8f} | "
            f"Val Loss: {val_loss:.8f} | "
            f"Val AUC: {val_auc:.8f}"
        )

        # Early Stopping & Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with AUC: {best_auc:.8f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.2f} seconds.")
    print(f"Best Validation AUC: {best_auc:.16f}")

    return best_model_path
