import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from library.config import Config
from library.utils import seed_everything, save_checkpoint, load_checkpoint
from library.dataset import get_dataloaders
from library.model_components import (
    GatedTransformerResFunnelHybrid,
    train_one_epoch,
    validate,
)


def train_model(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    debug=False,
    debug_samples=Config.DEBUG_SAMPLES,
):
    """
    Executes the training pipeline with configurable hyperparameters.

    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for DataLoaders.
        learning_rate (float): Initial learning rate.
        weight_decay (float): Weight decay for AdamW.
        debug (bool): If True, limits the dataset size for faster debugging.
        debug_samples (int): Number of samples to use in debug mode.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Starting training on {device}...")

    # Data Loading
    train_loader, val_loader, _, _ = get_dataloaders(
        batch_size=batch_size, load_cached_data=True
    )

    # Debug Slicing: Create subsets if debug mode is active
    if debug:
        print(
            f"Debug mode enabled. Limiting training/validation to {debug_samples} samples."
        )

        # Create subsets
        train_indices = range(min(len(train_loader.dataset), debug_samples))
        val_indices = range(min(len(val_loader.dataset), debug_samples))

        train_subset = Subset(train_loader.dataset, train_indices)
        val_subset = Subset(val_loader.dataset, val_indices)

        # Re-create loaders with subsets
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            drop_last=False,
        )

    # Model Initialization
    model = GatedTransformerResFunnelHybrid().to(device)

    # Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.SCHEDULER_STEP_SIZE, gamma=Config.SCHEDULER_GAMMA
    )

    criterion = nn.BCEWithLogitsLoss()

    # Training Loop
    best_auc = 0.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Print metrics with full precision (no rounding)
        print(
            f"Epoch {epoch}/{epochs} | LR: {current_lr} | "
            f"Train Loss: {train_loss} | Train AUC: {train_auc} | "
            f"Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Checkpoint based on Validation AUC
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_auc, Config.MODEL_CHECKPOINT
            )
            print(f"  >>> New Best Model Saved! AUC: {best_auc}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    print(f"Training complete. Best Validation AUC: {best_auc}")


def generate_submission(batch_size=Config.BATCH_SIZE):
    """
    Generates predictions for the test set using the best saved model.

    Args:
        batch_size (int): Batch size for inference.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print("Generating submission...")

    # Data
    _, _, test_loader, test_ids = get_dataloaders(
        batch_size=batch_size, load_cached_data=True
    )

    # Model
    model = GatedTransformerResFunnelHybrid().to(device)

    # Load Weights
    checkpoint = load_checkpoint(Config.MODEL_CHECKPOINT, model, device=device)
    print(
        f"Loaded model from epoch {checkpoint['epoch']} with Val AUC: {checkpoint['score']}"
    )

    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            cont = batch["cont_features"].to(device)
            cat = batch["cat_features"].to(device)

            logits = model(cont, cat)
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())

    all_preds = np.concatenate(all_preds).flatten()

    # Create DataFrame
    submission_df = pd.DataFrame({"id": test_ids, "target": all_preds})

    # Save
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
