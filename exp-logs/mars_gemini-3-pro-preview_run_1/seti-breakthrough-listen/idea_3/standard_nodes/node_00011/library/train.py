import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import Config
from library.utils import set_seed, AverageMeter, calculate_roc_auc, save_checkpoint
from library.model import CadenceModel2D
from library.data import get_dataloaders


def train_model(debug=Config.DEBUG, epochs=Config.EPOCHS):
    """
    Orchestrates the training of the Spatiotemporal 3D ResNet model.

    Args:
        debug (bool): If True, uses a smaller subset of data for debugging.
        epochs (int): Maximum number of training epochs.

    Returns:
        float: The best validation ROC AUC score achieved.
    """
    # Ensure reproducibility
    set_seed(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Initializing training on device: {device}")

    # 1. Data Loading
    # We use the get_dataloaders function from library.data which handles
    # the SETIDataset, augmentations, and reshaping to (1, 6, 273, 256).
    train_loader, val_loader, _ = get_dataloaders(
        debug=debug, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    # 2. Model Initialization
    # Load the 2D ResNet-34 with 6 input channels
    model = CadenceModel2D(pretrained=True)
    model = model.to(device)

    # 3. Optimization Setup
    # AdamW optimizer as specified in the idea
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Binary Cross Entropy with Logits Loss
    criterion = nn.BCEWithLogitsLoss()

    # OneCycleLR Scheduler
    # Steps per epoch is required for OneCycleLR
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.MAX_LR,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.3,
    )

    # 4. Training Loop
    best_score = 0.0
    patience_counter = 0

    print("Starting training loop...")

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_loss_meter = AverageMeter()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)  # Shape: (B, 1)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss_meter.update(loss.item(), inputs.size(0))

        # --- Validation Phase ---
        model.eval()
        val_loss_meter = AverageMeter()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).unsqueeze(1)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss_meter.update(loss.item(), inputs.size(0))

                # Apply sigmoid to get probabilities for AUC calculation
                probs = torch.sigmoid(outputs)

                # Collect predictions and targets
                all_preds.append(probs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        # Concatenate all batches
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        # Calculate Metric
        val_auc = calculate_roc_auc(all_targets, all_preds)

        # Print metrics (Full precision as requested)
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss_meter.avg}")
        print(f"Val Loss: {val_loss_meter.avg}")
        print(f"Val AUC: {val_auc}")

        # --- Checkpointing & Early Stopping ---
        if val_auc > best_score:
            best_score = val_auc
            print(
                f"New best model found! Saving checkpoint to {Config.MODEL_SAVE_PATH}"
            )
            save_checkpoint(model, optimizer, scheduler, epoch, best_score)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        # Flush stdout to ensure logs are visible immediately
        sys.stdout.flush()

    print(f"Training finished. Best Val AUC: {best_score}")
    return best_score
