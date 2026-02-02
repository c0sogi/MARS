import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

# Import from library
from library.config import Config
from library.utils import set_seed, save_checkpoint
from library.data import (
    get_loaders,
    load_and_process_json,
    IcebergDataset,
    get_transforms,
)
from library.model import IcebergResNet18
from library.calibration import PlattScaler
from library.engine import train_one_epoch, evaluate, update_bn, predict_tta


def run():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading data...")
    # Load raw data directly to handle CV splitting
    train_data_full = load_and_process_json(
        Config.TRAIN_JSON, "train", load_cached_data=True
    )
    test_loader = get_loaders(load_cached_data=True)[2]  # Keep test loader

    X_full = train_data_full["images"]
    ang_full = train_data_full["angles"]
    y_full = train_data_full["labels"]
    id_full = train_data_full["ids"]

    # 3. Phase 1: Global Epoch Selection via 5-Fold CV (Cite 00049, 00040)
    print("\n=== Phase 1: Global Epoch Selection (5-Fold CV) ===")

    kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)

    # Track validation loss per epoch for each fold
    fold_val_losses = np.zeros((Config.N_FOLDS, Config.MAX_EPOCHS_PHASE_1))
    oof_logits = np.zeros(len(y_full))
    oof_targets = np.zeros(len(y_full))

    # Store OOF predictions per epoch to find the best one later
    # Shape: (Epochs, N_Samples)
    oof_preds_history = np.zeros((Config.MAX_EPOCHS_PHASE_1, len(y_full)))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_full)):
        print(f"\n--- Fold {fold + 1}/{Config.N_FOLDS} ---")

        # Create Datasets
        train_ds = IcebergDataset(
            X_full[train_idx],
            ang_full[train_idx],
            y_full[train_idx],
            id_full[train_idx],
            transform=get_transforms("train"),
        )
        val_ds = IcebergDataset(
            X_full[val_idx],
            ang_full[val_idx],
            y_full[val_idx],
            id_full[val_idx],
            transform=get_transforms("val"),
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Init Model
        model = IcebergResNet18().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
        )

        for epoch in range(1, Config.MAX_EPOCHS_PHASE_1 + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
            val_loss, logits, _ = evaluate(model, val_loader, device)

            scheduler.step(val_loss)

            # Record metrics
            fold_val_losses[fold, epoch - 1] = val_loss

            # Store OOF predictions for this epoch
            # Note: evaluate returns logits
            oof_preds_history[epoch - 1, val_idx] = logits.flatten()

    # Calculate Average Validation Loss Curve
    avg_val_losses = fold_val_losses.mean(axis=0)
    best_epoch_idx = np.argmin(avg_val_losses)
    best_epoch = best_epoch_idx + 1
    min_avg_loss = avg_val_losses[best_epoch_idx]

    print(f"\nGlobal Optimal Epoch: {best_epoch} with Avg Val Loss: {min_avg_loss:.4f}")

    # Use OOF predictions from the best epoch for calibration and analysis
    final_oof_logits = oof_preds_history[best_epoch_idx]
    final_oof_targets = y_full

    # Calculate Log Loss on OOF
    # We use BCEWithLogitsLoss equivalent for metric
    # But for final reporting we need probabilities
    final_oof_probs = 1 / (1 + np.exp(-final_oof_logits))
    final_val_loss = log_loss(final_oof_targets, final_oof_probs)
    print(f"Final OOF Log Loss at Epoch {best_epoch}: {final_val_loss:.4f}")

    # 4. Phase 2: Calibration
    print("\n=== Phase 2: Meta-Calibration ===")
    scaler = PlattScaler()
    scaler.fit(final_oof_logits, final_oof_targets)
    scaler.save()

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Correlation between angle and loss
    val_probs = scaler.predict_proba(final_oof_logits)
    epsilon = 1e-15
    val_probs_clipped = np.clip(val_probs, epsilon, 1 - epsilon)
    sample_losses = -(
        final_oof_targets * np.log(val_probs_clipped)
        + (1 - final_oof_targets) * np.log(1 - val_probs_clipped)
    )

    # Handle NaNs in angles if any (though imputed in loading)
    valid_mask = ~np.isnan(ang_full)
    correlation = np.corrcoef(ang_full[valid_mask], sample_losses[valid_mask])[0, 1]
    print(f"Correlation between Incidence Angle and Log Loss: {correlation:.4f}")

    # 6. Submission Logic
    THRESHOLD = 0.16918645240183008
    if final_val_loss < THRESHOLD:
        print(
            f"\nValidation Metric {final_val_loss} < {THRESHOLD}. Proceeding to Phase 3 & Submission."
        )

        # 7. Phase 3: Production (Ensemble Training on Full Data)
        print("\n=== Phase 3: Production (SWA Ensemble) ===")

        # Full Dataset
        full_ds = IcebergDataset(
            X_full, ang_full, y_full, id_full, transform=get_transforms("train")
        )
        full_loader = DataLoader(
            full_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        ensemble_logits = []

        for i in range(Config.ENSEMBLE_SIZE):
            print(f"\nTraining Ensemble Model {i+1}/{Config.ENSEMBLE_SIZE}...")

            # Seed diversity
            current_seed = Config.SEED + i
            set_seed(current_seed)

            model = IcebergResNet18().to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )
            # Use same scheduler logic as Phase 1 for consistency
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=Config.SCHEDULER_FACTOR,
                patience=Config.SCHEDULER_PATIENCE,
            )

            # Train for best_epoch
            for epoch in range(1, best_epoch + 1):
                train_loss = train_one_epoch(
                    model, full_loader, optimizer, device, epoch
                )
                # Note: We don't have val loss to step the scheduler.
                # However, with 1e-4 LR and limited epochs, constant LR is often sufficient.
                # Or we can use a dummy step if we want to strictly follow code paths, but here we just train.

            # SWA Phase
            print(f"Starting SWA for Model {i+1}...")
            swa_model = AveragedModel(model).to(device)
            swa_scheduler = SWALR(optimizer, swa_lr=Config.SWA_LR)

            for swa_epoch in range(Config.SWA_DURATION):
                train_one_epoch(
                    model, full_loader, optimizer, device, f"SWA-{swa_epoch}"
                )
                swa_model.update_parameters(model)
                swa_scheduler.step()

            update_bn(full_loader, swa_model, device)

            # Inference
            logits, test_ids = predict_tta(swa_model, test_loader, device)
            ensemble_logits.append(logits)

        # Average Logits
        avg_logits = np.mean(ensemble_logits, axis=0)

        # Calibrate
        test_probs = scaler.predict_proba(avg_logits)

        # Save
        submission_df = pd.DataFrame({"id": test_ids, "is_iceberg": test_probs})
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation Metric {final_val_loss} >= {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    run()
