import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

# Import from provided libraries
from library.config import Config
from library.data import load_dataset_arrays, PathologyDataset, get_transforms
from library.models import get_model
from library.utils import seed_everything, ModelEMA, save_checkpoint, load_checkpoint
from library.engine import train_one_epoch, validate, predict


def main():
    # --- 1. Configuration & Setup ---
    seed_everything(Config.SEED)
    Config.setup()

    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Data Loading ---
    print("Loading data...")
    # Load Train Data (used for Cross-Validation)
    train_images, train_labels, train_ids = load_dataset_arrays(
        Config.TRAIN_METADATA_PATH, "train", load_cached_data=True
    )

    # Load Validation Data (Hold-out for final evaluation)
    val_images, val_labels, val_ids = load_dataset_arrays(
        Config.VAL_METADATA_PATH, "val", load_cached_data=True
    )

    # Load Test Data
    test_images, test_labels, test_ids = load_dataset_arrays(
        Config.TEST_METADATA_PATH, "test", load_cached_data=True
    )

    # --- 3. Training Loop (5-Fold CV x 2 Models) ---
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Store paths to best checkpoints
    checkpoint_paths = []

    for fold, (train_idx, inner_val_idx) in enumerate(
        skf.split(train_images, train_labels)
    ):
        print(f"\n{'='*20} Fold {fold+1}/{Config.NUM_FOLDS} {'='*20}")

        # Create Fold Datasets
        fold_train_images = train_images[train_idx]
        fold_train_labels = train_labels[train_idx]
        fold_train_ids = train_ids[train_idx]

        # Inner validation set for model selection/early stopping
        fold_val_images = train_images[inner_val_idx]
        fold_val_labels = train_labels[inner_val_idx]
        fold_val_ids = train_ids[inner_val_idx]

        train_dataset = PathologyDataset(
            fold_train_images,
            fold_train_labels,
            fold_train_ids,
            transforms=get_transforms("train"),
        )
        val_dataset = PathologyDataset(
            fold_val_images,
            fold_val_labels,
            fold_val_ids,
            transforms=get_transforms("val"),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Train each architecture in the heterogeneous ensemble
        for model_name in Config.MODEL_NAMES:
            print(f"\nTraining {model_name} on Fold {fold+1}...")

            model = get_model(model_name, pretrained=True).to(device)

            # Initialize EMA model
            ema_model = ModelEMA(model, device=device)

            optimizer = optim.AdamW(
                model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
            )

            best_auc = 0.0
            best_ckpt_path = os.path.join(
                Config.CHECKPOINT_DIR, f"best_{model_name}_fold_{fold}.pth"
            )

            for epoch in range(Config.EPOCHS):
                # Train
                train_one_epoch(
                    model, train_loader, optimizer, device, epoch, ema_model
                )

                # Validate using EMA weights
                auc = validate(ema_model.module, val_loader, device)

                scheduler.step()

                if auc > best_auc:
                    best_auc = auc
                    save_checkpoint(
                        ema_model.module,
                        optimizer,
                        scheduler,
                        epoch,
                        auc,
                        best_ckpt_path,
                    )
                    print(f"  New Best AUC: {best_auc:.5f}")

            checkpoint_paths.append(best_ckpt_path)

            # Cleanup to save memory
            del model, ema_model, optimizer, scheduler
            torch.cuda.empty_cache()

    # --- 4. Validation on Hold-out Set (Ensemble) ---
    print("\nRunning Ensemble Validation on Hold-out Set...")

    # Create Hold-out Loader
    holdout_dataset = PathologyDataset(
        val_images,
        val_labels,
        val_ids,
        transforms=get_transforms("test"),  # Use deterministic test transforms
    )
    holdout_loader = DataLoader(
        holdout_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Accumulate predictions from all models
    ensemble_preds_val = np.zeros(len(val_ids))

    for ckpt_path in checkpoint_paths:
        # Determine architecture from filename
        model_name = "convnext_tiny"

        print(f"Inference with {os.path.basename(ckpt_path)}...")
        model = get_model(model_name, pretrained=False)
        load_checkpoint(ckpt_path, model, device=device)
        model.to(device)

        # predict() uses 8-view TTA
        ids, preds = predict(model, holdout_loader, device)
        ensemble_preds_val += np.array(preds)

        del model
        torch.cuda.empty_cache()

    # Average predictions
    ensemble_preds_val /= len(checkpoint_paths)

    final_val_auc = roc_auc_score(val_labels, ensemble_preds_val)
    print(f"Final Validation Metric: {final_val_auc}")

    # --- 5. Failure Analysis ---
    print("\nPerforming Failure Analysis...")
    # Calculate absolute error
    errors = np.abs(val_labels - ensemble_preds_val)

    # Calculate simple image stats for validation set (normalized 0-1)
    val_imgs_norm = val_images.astype(np.float32) / 255.0

    # Brightness: Mean of all channels
    brightness = val_imgs_norm.mean(axis=(1, 2, 3))
    # Contrast: Std of all channels
    contrast = val_imgs_norm.std(axis=(1, 2, 3))
    # Red Mean: Mean of Red channel (index 0)
    red_mean = val_imgs_norm[:, :, :, 0].mean(axis=(1, 2))

    # Correlations
    corr_brightness = np.corrcoef(errors, brightness)[0, 1]
    corr_contrast = np.corrcoef(errors, contrast)[0, 1]
    corr_red = np.corrcoef(errors, red_mean)[0, 1]

    print(f"Correlation (Error vs Brightness): {corr_brightness:.4f}")
    print(f"Correlation (Error vs Contrast):   {corr_contrast:.4f}")
    print(f"Correlation (Error vs Red Mean):   {corr_red:.4f}")

    # --- 6. Test Inference & Submission ---
    threshold = 0.9889066475479729
    if final_val_auc > threshold:
        print("\nValidation metric meets threshold. Generating submission...")

        test_dataset = PathologyDataset(
            test_images, test_labels, test_ids, transforms=get_transforms("test")
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        ensemble_preds_test = np.zeros(len(test_ids))
        test_ids_list = []

        for i, ckpt_path in enumerate(checkpoint_paths):
            model_name = "convnext_tiny"

            print(f"Test Inference with {os.path.basename(ckpt_path)}...")
            model = get_model(model_name, pretrained=False)
            load_checkpoint(ckpt_path, model, device=device)
            model.to(device)

            ids, preds = predict(model, test_loader, device)
            ensemble_preds_test += np.array(preds)

            if i == 0:
                test_ids_list = ids

            del model
            torch.cuda.empty_cache()

        ensemble_preds_test /= len(checkpoint_paths)

        # Create submission DataFrame
        df_sub = pd.DataFrame({"id": test_ids_list, "label": ensemble_preds_test})

        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation metric {final_val_auc} did not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
