import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from library import config
from library import utils
from library import data_loader
from library import model as lib_model


def run_training(
    num_epochs=config.NUM_EPOCHS,
    batch_size=config.BATCH_SIZE,
    learning_rate=config.LEARNING_RATE,
    weight_decay=config.WEIGHT_DECAY,
    early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
    debug=False,
    load_cached_data=True,
):
    """
    Orchestrates the training, validation, and inference pipeline.

    Args:
        num_epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for DataLoaders.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        early_stopping_patience (int): Epochs to wait for improvement before stopping.
        debug (bool): If True, runs on a small subset of data for 1 epoch.
        load_cached_data (bool): Whether to load ROI cache from disk.
    """

    # 1. Setup
    utils.set_seed(config.SEED)
    device = utils.get_device()
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = data_loader.get_dataloaders(
        load_cached_data=load_cached_data
    )

    # Debug Mode: Truncate datasets to verify pipeline quickly
    if debug:
        print("DEBUG MODE: Truncating datasets...")
        limit = 32  # Small number of samples

        # Helper to slice dataset
        def truncate_dataset(loader):
            if hasattr(loader.dataset, "metadata"):
                loader.dataset.metadata = loader.dataset.metadata.iloc[:limit]
                # Reset the length cache of the loader if necessary,
                # though usually creating a new loader is cleaner.
                # Since we can't easily recreate the loader with the same params
                # without duplicating code, we rely on the dataset modification.

        truncate_dataset(train_loader)
        truncate_dataset(val_loader)
        truncate_dataset(test_loader)

        num_epochs = 1
        print(
            f"Debug mode enabled: Running for {num_epochs} epoch with {limit} samples."
        )

    # 3. Model Initialization
    print("Initializing AsymmetricEfficientNet...")
    model = lib_model.AsymmetricEfficientNet().to(device)

    # 4. Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # 5. Training Loop
    best_auc = 0.0
    patience_counter = 0

    print("Starting training...")
    for epoch in range(num_epochs):
        # Train
        train_loss = lib_model.train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_auc = lib_model.validate(model, val_loader, criterion, device)

        # Logging (Full precision)
        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {train_loss} - "
            f"Val Loss: {val_loss} - "
            f"Val AUC: {val_auc}"
        )

        # Checkpointing & Early Stopping
        is_best = val_auc > best_auc
        if is_best:
            best_auc = val_auc
            patience_counter = 0
            utils.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_auc": best_auc,
                    "optimizer": optimizer.state_dict(),
                },
                is_best=True,
                checkpoint_dir=config.CACHE_DIR,
            )
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Best Validation AUC: {best_auc}")

    # 6. Inference
    print("Loading best model for inference...")
    best_model_path = os.path.join(config.CACHE_DIR, "best_model.pth")

    # Ensure best model exists (if training failed or 0 epochs, use current)
    if os.path.exists(best_model_path):
        utils.load_checkpoint(model, path=best_model_path, device=device)
    else:
        print("Warning: No best model checkpoint found. Using current model state.")

    lib_model.predict_and_submit(model, test_loader, device, config.SUBMISSION_PATH)
