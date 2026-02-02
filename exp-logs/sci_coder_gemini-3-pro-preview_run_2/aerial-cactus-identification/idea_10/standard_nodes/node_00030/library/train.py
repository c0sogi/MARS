import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from library.config import (
    SEEDS,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
    set_seed,
)
from library.dataset import get_dataloaders
from library.model import (
    CustomNarrowSEMultiScaleResNet,
    train_one_epoch,
    validate,
    generate_submission,
)
from library.utils import get_device, save_checkpoint


def train_single_seed(seed, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE):
    """
    Trains a single model instance for a specific seed.

    Args:
        seed (int): Random seed for reproducibility.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for data loaders.

    Returns:
        float: Best validation AUC achieved.
    """
    set_seed(seed)
    device = get_device()

    # DataLoaders
    # We only need train and val loaders for the training loop
    train_loader, val_loader, _, _ = get_dataloaders(
        batch_size=batch_size, num_workers=4, load_cached_data=True
    )

    # Model Initialization
    model = CustomNarrowSEMultiScaleResNet().to(device)

    # Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_auc = 0.0
    patience_counter = 0

    print(f"\nStarting training for Seed {seed}...")

    for epoch in range(epochs):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        scheduler.step()

        # Print metrics with full precision as requested
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss} | Train AUC: {train_auc} | "
            f"Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Checkpoint and Early Stopping Logic
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            save_checkpoint(model.state_dict(), f"model_seed_{seed}.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Seed {seed} completed. Best Val AUC: {best_val_auc}")
    return best_val_auc


def run_training(epochs=NUM_EPOCHS):
    """
    Runs the training pipeline for all configured seeds.
    """
    val_aucs = []
    for seed in SEEDS:
        auc = train_single_seed(seed, epochs=epochs)
        val_aucs.append(auc)

    print(f"\nTraining Complete. Average Best Val AUC: {np.mean(val_aucs)}")


def run_submission():
    """
    Generates submission using the trained models.
    """
    generate_submission()
