import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import provided library modules
from library.config import Config
from library.utils import (
    seed_everything,
    calculate_auc,
    save_checkpoint,
    write_submission,
    AverageMeter,
)
from library.data import get_dataloaders, get_test_dataloader
from library.models import get_model
from library.engine import train_one_epoch, SWAManager


def main():
    # 1. Setup
    # Instantiate Config with reduced epochs to ensure completion within time limits
    # while maintaining sufficient training for convergence on small data.
    config = Config(epochs=30)
    seed_everything(config.SEED)
    device = torch.device(config.DEVICE)

    print(f"Starting execution with {config.EPOCHS} epochs per model.")
    print(f"Models: {config.MODEL_BACKBONES}")
    print(f"Folds: {config.NUM_FOLDS}")

    # Storage for Ensemble Aggregation
    # val_storage[fold_idx] = {'targets': [], 'preds': np.array, 'features': []}
    # This dictionary aggregates predictions from different models for the same fold.
    val_storage = {}

    # Test predictions accumulator (sum of probabilities)
    test_preds_sum = None
    test_ids = None

    # Load Test Loader (Shared across all runs)
    test_loader, test_ids_loaded = get_test_dataloader(load_cached_data=True)
    test_ids = test_ids_loaded

    # 2. Training Loop
    total_models_trained = 0

    for model_name in config.MODEL_BACKBONES:
        for fold in range(config.NUM_FOLDS):
            print(f"\n=== Training {model_name} | Fold {fold} ===")

            # Load Data
            train_loader, val_loader = get_dataloaders(fold=fold, load_cached_data=True)

            # Initialize Model
            model = get_model(model_name, pretrained=True)
            model = model.to(device)

            # Optimizer & Scheduler
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config.LEARNING_RATE,
                weight_decay=config.WEIGHT_DECAY,
            )

            # SWA Scheduler Setup
            # Cosine Annealing for the first 75% of epochs
            swa_start = int(config.EPOCHS * config.SWA_START_EPOCH_PCT)
            scheduler = CosineAnnealingLR(optimizer, T_max=swa_start)
            # SWA Manager handles the switch to constant LR and weight averaging
            swa_manager = SWAManager(model, optimizer)

            # Loss
            criterion = nn.BCEWithLogitsLoss()

            # Train
            for epoch in range(config.EPOCHS):
                avg_loss = train_one_epoch(
                    model, train_loader, criterion, optimizer, device, epoch
                )
                swa_manager.step(epoch, model, base_scheduler=scheduler)

            # Finalize SWA
            # Updates Batch Normalization statistics using the training data
            final_model = swa_manager.finalize(train_loader, device)
            final_model.eval()

            # Save Checkpoint
            ckpt_path = os.path.join(
                config.CHECKPOINT_DIR, f"{model_name}_fold_{fold}.pth"
            )
            save_checkpoint(final_model.state_dict(), ckpt_path)

            # --- Validation Inference ---
            fold_targets = []
            fold_preds = []
            fold_features = []  # Store (brightness, contrast) for failure analysis

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)

                    # Forward pass
                    outputs = final_model(images)
                    probs = torch.sigmoid(outputs).cpu().numpy()

                    fold_preds.append(probs)

                    # Only need to store targets/features once per fold (from the first model encountered)
                    # because the validation set is identical for a given fold index.
                    if fold not in val_storage:
                        fold_targets.append(labels.cpu().numpy())

                        # Compute features for failure analysis
                        # images: (B, 3, H, W)
                        imgs_np = images.cpu().numpy()
                        # Mean over H, W, C
                        b = imgs_np.mean(axis=(1, 2, 3))
                        c = imgs_np.std(axis=(1, 2, 3))
                        feats = np.stack([b, c], axis=1)
                        fold_features.append(feats)

            fold_preds = np.concatenate(fold_preds, axis=0)

            # Aggregate Validation Predictions
            if fold not in val_storage:
                val_storage[fold] = {
                    "targets": np.concatenate(fold_targets, axis=0),
                    "features": np.concatenate(fold_features, axis=0),
                    "preds": fold_preds,
                }
            else:
                # Add to existing predictions (Ensemble averaging)
                val_storage[fold]["preds"] += fold_preds

            # --- Test Inference ---
            fold_test_preds = []
            with torch.no_grad():
                for images, _ in test_loader:
                    images = images.to(device)
                    outputs = final_model(images)
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    fold_test_preds.append(probs)

            fold_test_preds = np.concatenate(fold_test_preds, axis=0)

            if test_preds_sum is None:
                test_preds_sum = fold_test_preds
            else:
                test_preds_sum += fold_test_preds

            total_models_trained += 1

    # 3. Post-Processing & Evaluation

    # Average Test Predictions
    test_preds_avg = test_preds_sum / total_models_trained

    # Process Validation Data
    all_val_targets = []
    all_val_preds = []
    all_val_features = []

    num_models_per_fold = len(config.MODEL_BACKBONES)

    # Concatenate data from all folds
    for fold in sorted(val_storage.keys()):
        data = val_storage[fold]
        # Average predictions for this fold (divide by number of models)
        avg_preds = data["preds"] / num_models_per_fold

        all_val_targets.append(data["targets"])
        all_val_preds.append(avg_preds)
        all_val_features.append(data["features"])

    all_val_targets = np.concatenate(all_val_targets, axis=0)
    all_val_preds = np.concatenate(all_val_preds, axis=0)
    all_val_features = np.concatenate(all_val_features, axis=0)

    # Calculate Metric
    final_auc = calculate_auc(all_val_targets, all_val_preds)
    print(f"Final Validation Metric: {final_auc}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate per-sample loss (Binary Cross Entropy averaged over classes)
    eps = 1e-7
    preds_clipped = np.clip(all_val_preds, eps, 1 - eps)
    # BCE = - (y * log(p) + (1-y) * log(1-p))
    # We compute mean over classes (axis 1) to get a scalar error metric per sample
    sample_losses = -(
        all_val_targets * np.log(preds_clipped)
        + (1 - all_val_targets) * np.log(1 - preds_clipped)
    )
    mean_sample_loss = sample_losses.mean(axis=1)

    # Extract Features
    brightness = all_val_features[:, 0]
    contrast = all_val_features[:, 1]
    label_count = all_val_targets.sum(axis=1)

    # Calculate Correlations
    corr_brightness = np.corrcoef(mean_sample_loss, brightness)[0, 1]
    corr_contrast = np.corrcoef(mean_sample_loss, contrast)[0, 1]
    corr_complexity = np.corrcoef(mean_sample_loss, label_count)[0, 1]

    print(f"Correlation (Error vs Brightness): {corr_brightness:.4f}")
    print(f"Correlation (Error vs Contrast): {corr_contrast:.4f}")
    print(f"Correlation (Error vs Label Count): {corr_complexity:.4f}")

    # 5. Submission
    THRESHOLD = 0.9479806884980326
    if final_auc > THRESHOLD:
        print("\nMetric threshold met. Generating submission...")

        submission_ids = []
        submission_probs = []

        # Format: Id = rec_id * 100 + species_idx
        for i, rec_id in enumerate(test_ids):
            probs = test_preds_avg[i]
            for species_idx, prob in enumerate(probs):
                sub_id = int(rec_id * 100 + species_idx)
                submission_ids.append(sub_id)
                submission_probs.append(prob)

        write_submission(submission_ids, submission_probs)
    else:
        print(
            f"\nMetric {final_auc} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
