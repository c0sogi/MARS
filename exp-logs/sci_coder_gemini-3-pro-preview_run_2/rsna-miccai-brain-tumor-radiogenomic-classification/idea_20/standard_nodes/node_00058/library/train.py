import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from library.config import Config
from library.utils import (
    AverageMeter,
    calculate_roc_auc,
    save_checkpoint,
    set_seed,
    get_device,
)
from library.data_loader import get_dataloaders
from library.model import AsymmetricEfficientNet


def train_one_epoch(train_loader, model, criterion, optimizer, device, epoch):
    """
    Handles the training of one epoch.
    """
    model.train()
    losses = AverageMeter()

    for i, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)  # (B, 1)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        losses.update(loss.item(), images.size(0))

    print(f"Epoch [{epoch}/{Config.NUM_EPOCHS}] Training Loss: {losses.avg}")
    return losses.avg


def validate(val_loader, model, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC score.
    """
    model.eval()
    losses = AverageMeter()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, targets)

            # Apply sigmoid for probability calculation used in AUC
            probs = torch.sigmoid(outputs)

            losses.update(loss.item(), images.size(0))
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(probs.cpu().numpy())

    # Calculate AUC
    # Flatten arrays
    all_targets = np.array(all_targets).flatten()
    all_preds = np.array(all_preds).flatten()

    auc_score = calculate_roc_auc(all_targets, all_preds)

    print(f"Validation Loss: {losses.avg}")
    print(f"Validation ROC AUC: {auc_score}")

    return losses.avg, auc_score


def run_training(load_cached_data=True):
    """
    Main execution function for the training pipeline.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = get_device()
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=load_cached_data)

    # 3. Model Initialization
    print("Initializing Asymmetric Grouped EfficientNet...")
    model = AsymmetricEfficientNet()
    model = model.to(device)

    # 4. Optimization Setup
    # AdamW with aggressive weight decay and low LR as per design
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            train_loader, model, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_auc = validate(val_loader, model, criterion, device)

        epoch_duration = time.time() - start_time
        print(f"Epoch {epoch} completed in {epoch_duration} seconds.")

        # Checkpoint & Early Stopping Logic
        # We save based on Validation Loss as specified
        is_best = val_loss < best_val_loss

        if is_best:
            print(
                f"Validation Loss improved from {best_val_loss} to {val_loss}. Saving checkpoint."
            )
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            state = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "val_auc": val_auc,
            }
            save_checkpoint(state, is_best=True, filepath=Config.MODEL_CHECKPOINT_PATH)
        else:
            patience_counter += 1
            print(
                f"Validation Loss did not improve. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        # Early Stopping
        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered. Training finished.")
            break

    print(f"Training complete. Best Validation Loss: {best_val_loss}")
