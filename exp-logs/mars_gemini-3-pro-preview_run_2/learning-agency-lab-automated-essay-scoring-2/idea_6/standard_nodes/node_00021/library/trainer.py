import os
import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup

from library.config import Config
from library.utils import AWP
from library.model_semantic import (
    DebertaV3Regressor,
    train_one_epoch,
    validate_one_epoch,
)

# Alias functions as requested by the task description
train_fn = train_one_epoch
valid_fn = validate_one_epoch


def run_fold(fold_idx, train_loader, val_loader):
    """
    Orchestrates the training process for a single fold of the semantic model.

    Args:
        fold_idx (int): Index of the current fold.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.

    Returns:
        float: The best Quadratic Weighted Kappa (QWK) score achieved on the validation set.
    """
    print(f"\n=== Training Semantic Model | Fold {fold_idx} ===")

    device = Config.DEVICE

    # Initialize Model
    model = DebertaV3Regressor()
    model.to(device)

    # Initialize Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Initialize Scheduler
    num_train_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_train_steps),
        num_training_steps=num_train_steps,
    )

    # Initialize Loss Function
    if Config.LOSS_FN == "SmoothL1Loss":
        criterion = nn.SmoothL1Loss()
    else:
        criterion = nn.MSELoss()

    # Initialize Adversarial Weight Perturbation (AWP)
    awp = None
    if Config.USE_AWP:
        awp = AWP(
            model,
            optimizer,
            adv_lr=Config.AWP_LR,
            adv_eps=Config.AWP_EPS,
            start_epoch=Config.AWP_START_EPOCH,
        )

    # Initialize Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Training Loop State
    best_qwk = -1.0
    patience = 3
    early_stopping_counter = 0
    save_path = os.path.join(Config.MODEL_DIR, f"deberta_fold_{fold_idx}.bin")

    for epoch in range(Config.EPOCHS):
        # Train Step
        train_loss = train_fn(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            epoch,
            awp,
            scaler,
        )

        # Validation Step
        val_loss, val_preds, val_qwk = valid_fn(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val QWK: {val_qwk}"
        )

        # Save Best Model
        if val_qwk > best_qwk:
            best_qwk = val_qwk
            print(f"Score Improved. Saving model to {save_path}...")
            torch.save(model.state_dict(), save_path)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        # Early Stopping
        if early_stopping_counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs with no improvement."
            )
            break

    return best_qwk
