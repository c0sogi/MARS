import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import get_logger, seed_everything
from library.data_loader import get_dataloaders
from library.model import (
    ParallelFactorizedDCNResNet,
    train_one_epoch,
    validate,
    predict_test,
)

logger = get_logger()


def run_training(
    epochs: int = Config.EPOCHS,
    load_cached_data: bool = True,
    patience: int = Config.PATIENCE,
):
    """
    Orchestrates the training process, including data loading, model initialization,
    training loop with early stopping, and submission generation.

    Args:
        epochs (int): Maximum number of training epochs.
        load_cached_data (bool): Whether to load pre-processed data from cache.
        patience (int): Number of epochs to wait for improvement before early stopping.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    logger.info(f"Using device: {device}")

    # 2. Data Loading
    # get_dataloaders handles caching and preprocessing internally
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # Determine input dimensions dynamically from a batch
    dummy_X, _ = next(iter(train_loader))
    input_dim = dummy_X.shape[1]
    # Cover Type dataset has 7 classes (integers 1-7)
    num_classes = 7

    logger.info(f"Input Dimension: {input_dim}")
    logger.info(f"Num Classes: {num_classes}")

    # 3. Model Initialization
    model = ParallelFactorizedDCNResNet(
        input_dim=input_dim,
        num_classes=num_classes,
        dcn_rank=Config.DCN_RANK,
        hidden_dim=Config.HIDDEN_DIM,
        resnet_blocks=Config.RESNET_BLOCKS,
        dropout=Config.DROPOUT,
    ).to(device)

    # 4. Optimization Setup
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss()

    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=Config.SCHEDULER_ETA_MIN
    )

    # 5. Training Loop
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0

    logger.info("Starting training...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Train and Validate
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Log metrics with full precision
        logger.info(
            f"Epoch {epoch}/{epochs} | LR: {current_lr:.8f} | "
            f"Train Loss: {train_loss:.20f} | Train Acc: {train_acc:.20f} | "
            f"Val Loss: {val_loss:.20f} | Val Acc: {val_acc:.20f} | "
            f"Time: {time.time() - epoch_start:.2f}s"
        )

        # Early Stopping & Checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Deepcopy to ensure we save the exact weights, not a reference
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            logger.info(f"New best model found! Acc: {best_val_acc:.20f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

    total_time = time.time() - start_time
    logger.info(
        f"Training finished in {total_time:.2f}s. Best Val Acc: {best_val_acc:.20f}"
    )

    # 6. Prediction
    if best_model_state is not None:
        logger.info("Restoring best model weights for inference...")
        model.load_state_dict(best_model_state)

    logger.info("Generating predictions on Test set...")
    ids, preds = predict_test(model, test_loader, device)

    # 7. Submission
    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)

    logger.info(f"Saving submission to {Config.SUBMISSION_FILE}...")
    df_sub = pd.DataFrame({"Id": ids, "Cover_Type": preds})

    # Ensure strict integer types for submission
    df_sub["Id"] = df_sub["Id"].astype(int)
    df_sub["Cover_Type"] = df_sub["Cover_Type"].astype(int)

    df_sub.to_csv(Config.SUBMISSION_FILE, index=False)
    logger.info("Submission saved successfully.")
