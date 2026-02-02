import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import Config
from library.utils import (
    set_seed,
    save_checkpoint,
    load_checkpoint,
    save_submission,
)
from library.dataset import get_dataloaders
from library.model import (
    UltraWideECARepNeXt,
    train_one_epoch,
    validate,
    predict_tta,
)


def run_training(
    epochs=Config.EPOCHS,
    seeds=Config.SEEDS,
    batch_size=Config.BATCH_SIZE,
    debug=False,
):
    """
    Executes the training pipeline with Homogeneous Seed Averaging.

    Args:
        epochs (int): Number of training epochs per seed.
        seeds (list): List of seeds to run.
        batch_size (int): Batch size for data loaders.
        debug (bool): If True, runs on a small subset of data for debugging.
    """
    device = torch.device(Config.DEVICE)
    print(f"Running training on device: {device}")

    # 1. Data Loading
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)

    # 2. Debug Mode: Truncate datasets if requested
    if debug:
        print("DEBUG MODE: Truncating datasets to 100 samples.")
        limit = 100
        # Truncate Train
        train_loader.dataset.images = train_loader.dataset.images[:limit]
        train_loader.dataset.labels = train_loader.dataset.labels[:limit]
        train_loader.dataset.ids = train_loader.dataset.ids[:limit]
        # Truncate Val
        val_loader.dataset.images = val_loader.dataset.images[:limit]
        val_loader.dataset.labels = val_loader.dataset.labels[:limit]
        val_loader.dataset.ids = val_loader.dataset.ids[:limit]
        # Truncate Test
        test_loader.dataset.images = test_loader.dataset.images[:limit]
        test_loader.dataset.labels = test_loader.dataset.labels[:limit]
        test_loader.dataset.ids = test_loader.dataset.ids[:limit]

        # Adjust epochs for debug
        epochs = 2

    test_preds_accumulator = None
    test_ids = None

    # 3. Seed Loop
    for seed in seeds:
        print(f"\n--- Starting Seed {seed} ---")
        set_seed(seed)

        # Initialize Model
        model = UltraWideECARepNeXt().to(device)

        # Optimizer & Scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.BCEWithLogitsLoss()

        best_auc = 0.0
        patience_counter = 0
        model_save_path = Config.get_model_save_path(seed)

        # 4. Training Loop
        for epoch in range(epochs):
            train_loss, train_auc = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_auc = validate(model, val_loader, criterion, device)

            scheduler.step()

            # Print metrics (Full precision for validation as requested)
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | Train AUC: {train_auc:.6f} | "
                f"Val Loss: {val_loss} | Val AUC: {val_auc}"
            )

            # Early Stopping Check
            if val_auc > best_auc:
                best_auc = val_auc
                save_checkpoint(model.state_dict(), model_save_path)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

        # 5. Inference (Test Time Augmentation)
        print(f"Loading best model for seed {seed} (AUC: {best_auc})...")
        load_checkpoint(model, model_save_path, device)

        print("Switching to deploy mode (fusing weights)...")
        model.switch_to_deploy()

        ids, preds = predict_tta(model, test_loader, device)
        preds = np.array(preds)

        # Accumulate predictions
        if test_preds_accumulator is None:
            test_preds_accumulator = preds
            test_ids = ids
        else:
            test_preds_accumulator += preds

    # 6. Submission Generation
    if test_preds_accumulator is not None:
        final_preds = test_preds_accumulator / len(seeds)
        print(f"Saving submission to {Config.SUBMISSION_PATH}...")
        save_submission(test_ids, final_preds)
    else:
        print("No predictions generated.")
