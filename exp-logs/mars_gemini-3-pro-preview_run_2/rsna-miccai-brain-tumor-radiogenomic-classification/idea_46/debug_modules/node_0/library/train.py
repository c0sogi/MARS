import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import numpy as np

from library import config, data, model, utils


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device).view(-1, 1)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Store predictions for AUC calculation
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(probs)

    epoch_loss = running_loss / len(loader.dataset)

    # Calculate AUC safely (handle cases with only one class in batch)
    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    """
    Performs validation on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).view(-1, 1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(outputs).cpu().numpy()
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(probs)

    val_loss = running_loss / len(loader.dataset)

    try:
        val_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        val_auc = 0.5

    return val_loss, val_auc


def run_training(
    train_metadata_path=config.TRAIN_METADATA_PATH,
    val_metadata_path=config.VAL_METADATA_PATH,
    epochs=config.EPOCHS,
    patience=config.EARLY_STOPPING_PATIENCE,
):
    """
    Main driver function to run the training pipeline.
    """
    # Ensure reproducibility
    utils.set_seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize DataLoaders
    train_loader = data.get_dataloader(
        train_metadata_path, is_train=True, shuffle=True, batch_size=config.BATCH_SIZE
    )
    val_loader = data.get_dataloader(
        val_metadata_path, is_train=False, shuffle=False, batch_size=config.BATCH_SIZE
    )

    # Initialize Model
    net = model.AsymmetricEfficientNet().to(device)

    # Optimizer: AdamW with aggressive weight decay
    optimizer = optim.AdamW(
        net.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )

    # Loss Function: Binary Cross Entropy with Logits
    criterion = nn.BCEWithLogitsLoss()

    # Training State
    best_val_auc = -1.0
    patience_counter = 0

    print(f"Starting training for {epochs} epochs...")
    print("-" * 80)

    for epoch in range(epochs):
        start_time = time.time()

        # Train and Validate
        train_loss, train_auc = train_one_epoch(
            net, train_loader, optimizer, criterion, device
        )
        val_loss, val_auc = validate(net, val_loader, criterion, device)

        duration = time.time() - start_time

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} [{duration:.1f}s] - "
            f"Train Loss: {train_loss}, Train AUC: {train_auc} | "
            f"Val Loss: {val_loss}, Val AUC: {val_auc}"
        )

        # Checkpointing based on Validation AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0

            # Save the best model
            os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
            torch.save(net.state_dict(), config.MODEL_SAVE_PATH)
            print(f"  -> New best model saved (Val AUC: {val_auc})")
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience: {patience_counter}/{patience}")

        # Early Stopping
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print("-" * 80)
    print(f"Training complete. Best Validation AUC: {best_val_auc}")
