import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr

# Import from library
from library.config import Config
from library.utils import seed_everything, save_checkpoint, calculate_auc
from library.dataset import create_datasets, PathologyDataset, get_transforms
from library.model import ConvNeXtGeM
from library.engine import train_epoch, valid_epoch, inference_fn


def main():
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)

    # Override Config for Fast Baseline Execution
    # Reducing epochs to ensure 5-fold CV finishes within 2 hours
    Config.EPOCHS = 3
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading datasets...")
    # Load raw datasets (we will merge train and val for CV)
    train_ds_raw, val_ds_raw, test_ds = create_datasets(load_cached_data=True)

    # Merge arrays for Cross-Validation
    all_images = np.concatenate([train_ds_raw.images, val_ds_raw.images], axis=0)
    all_labels = np.concatenate([train_ds_raw.labels, val_ds_raw.labels], axis=0)

    print(f"Total merged samples: {len(all_labels)}")

    # 3. Cross-Validation Loop
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    oof_preds = np.zeros(len(all_labels))
    oof_targets = np.zeros(len(all_labels))
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_images, all_labels)):
        print(f"\n{'='*20} Fold {fold} {'='*20}")

        # Split Data
        X_train, X_val = all_images[train_idx], all_images[val_idx]
        y_train, y_val = all_labels[train_idx], all_labels[val_idx]

        # Create Datasets with appropriate transforms
        train_dataset = PathologyDataset(
            X_train, y_train, transforms=get_transforms("train")
        )
        val_dataset = PathologyDataset(X_val, y_val, transforms=get_transforms("val"))

        # Create Loaders
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

        # Initialize Model
        model = ConvNeXtGeM(config=Config)
        model.to(device)

        # Optimizer & Scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
        )

        # Training Loop
        best_auc = 0.0

        for epoch in range(Config.EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
            val_loss, val_auc = valid_epoch(model, val_loader, device)
            scheduler.step()

            is_best = val_auc > best_auc
            if is_best:
                best_auc = val_auc

            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_auc": best_auc,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                fold,
            )

        print(f"Fold {fold} Best AUC: {best_auc:.6f}")
        fold_aucs.append(best_auc)

        # Load best model for OOF predictions
        best_model_path = os.path.join(
            Config.CHECKPOINT_DIR, f"best_model_fold_{fold}.pth"
        )
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        # Generate predictions for validation set (no TTA for OOF to save time, or simple inference)
        # Using valid_epoch logic but extracting preds
        fold_preds = []
        fold_targets = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.sigmoid(outputs)
                fold_preds.extend(probs.cpu().numpy().flatten())
                fold_targets.extend(labels.numpy().flatten())

        oof_preds[val_idx] = fold_preds
        oof_targets[val_idx] = fold_targets

    # 4. Global Validation Metric
    final_auc = calculate_auc(oof_targets, oof_preds)
    print(f"\nFinal Validation Metric: {final_auc}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate error magnitude
    errors = np.abs(oof_targets - oof_preds)

    # Compute image stats for the whole dataset (to correlate with errors)
    # Note: We compute this on the raw images (before normalization/crop) to capture physical properties
    # Since we have all_images in memory, we can do this efficiently.
    # To save time, we'll do it on a subset or just iterate efficiently.

    print("Computing image statistics for correlation analysis...")
    brightness = []
    contrast = []
    red_mean = []

    # Iterate over all images (this matches oof_preds indices)
    # Using a simple loop or vectorized if possible.
    # all_images is (N, 96, 96, 3) uint8

    # Vectorized calculation for speed
    # Convert to float for stats
    imgs_float = all_images.astype(np.float32) / 255.0

    # Mean across H,W (spatial)
    spatial_means = np.mean(imgs_float, axis=(1, 2))  # Shape (N, 3)
    # Std across H,W
    spatial_stds = np.std(imgs_float, axis=(1, 2))  # Shape (N, 3)

    # Brightness: mean of channels
    brightness = np.mean(spatial_means, axis=1)

    # Contrast: mean of channel stds (approx)
    contrast = np.mean(spatial_stds, axis=1)

    # Red mean (Channel 0)
    red_mean = spatial_means[:, 0]

    # Calculate Correlations
    corr_brightness, _ = pearsonr(errors, brightness)
    corr_contrast, _ = pearsonr(errors, contrast)
    corr_red, _ = pearsonr(errors, red_mean)

    print(f"Correlation (Error vs Brightness): {corr_brightness:.4f}")
    print(f"Correlation (Error vs Contrast):   {corr_contrast:.4f}")
    print(f"Correlation (Error vs Red Mean):   {corr_red:.4f}")

    # 6. Submission Generation
    THRESHOLD = 0.9889066475479729

    if final_auc > THRESHOLD:
        print(
            f"\nValidation Metric ({final_auc}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        ensemble_preds = np.zeros(len(test_ds))

        for fold in range(Config.N_FOLDS):
            print(f"Inference with model fold {fold}...")
            model = ConvNeXtGeM(config=Config)
            model.to(device)

            # Load checkpoint
            ckpt_path = os.path.join(
                Config.CHECKPOINT_DIR, f"best_model_fold_{fold}.pth"
            )
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["state_dict"])

            # Inference with TTA
            preds = inference_fn(model, test_loader, device)
            ensemble_preds += preds

        # Average
        ensemble_preds /= Config.N_FOLDS

        # Create Submission DataFrame
        # We need IDs. The test dataset loads from test.csv which has IDs.
        # We can read test.csv to get IDs order.
        df_test = pd.read_csv(Config.TEST_META_PATH)
        submission = pd.DataFrame({"id": df_test["id"], "label": ensemble_preds})

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation Metric ({final_auc}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
