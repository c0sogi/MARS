import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import cv2

# Import from library
from library.config import Config
from library.utils import seed_everything, save_checkpoint, load_checkpoint
from library.dataset import load_dataset_metadata, get_transforms, PathologyDataset
from library.models import get_model
from library.engine import train_one_epoch, valid_one_epoch, tta_inference_fn
from library.stacking import (
    train_meta_learner,
    predict_with_meta_learner,
    create_submission,
)


def analyze_failures(val_df, preds, N=2000):
    """
    Performs failure analysis by correlating error with simple image statistics.
    """
    print("\n--- Failure Analysis ---")
    targets = val_df["label"].values
    errors = np.abs(targets - preds)

    # Select a subset for analysis to save time
    if len(val_df) > N:
        # Use a fixed seed for random choice to ensure reproducibility
        np.random.seed(Config.SEED)
        indices = np.random.choice(len(val_df), N, replace=False)
    else:
        indices = np.arange(len(val_df))

    subset_errors = errors[indices]

    # Calculate simple image stats
    brightness = []
    contrast = []

    print(f"Analyzing {len(indices)} validation samples for error correlation...")

    for idx in indices:
        row = val_df.iloc[idx]
        path = os.path.join(Config.INPUT_DIR, row["file_path"])
        img = cv2.imread(path)
        if img is None:
            b, c = 0, 0
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            b = np.mean(gray)
            c = np.std(gray)
        brightness.append(b)
        contrast.append(c)

    brightness = np.array(brightness)
    contrast = np.array(contrast)

    # Correlation
    # Handle edge case where std is 0 (constant images)
    if np.std(subset_errors) == 0:
        corr_b, corr_c = 0.0, 0.0
    else:
        corr_b = (
            np.corrcoef(subset_errors, brightness)[0, 1]
            if np.std(brightness) > 0
            else 0
        )
        corr_c = (
            np.corrcoef(subset_errors, contrast)[0, 1] if np.std(contrast) > 0 else 0
        )

    print(f"Correlation between Error and Brightness: {corr_b:.4f}")
    print(f"Correlation between Error and Contrast: {corr_c:.4f}")


def run():
    # 1. Setup
    seed_everything(Config.SEED)

    # Override Config for Fast Baseline to meet 2-hour limit
    # We use a stratified subset of the training data and fewer epochs.
    # 25,000 samples is sufficient for a strong baseline while being fast.
    TRAIN_SUBSET_SIZE = 25000
    EPOCHS = 5
    BATCH_SIZE = Config.BATCH_SIZE

    print(f"Starting Fast Baseline Run")
    print(f"Device: {Config.DEVICE}")
    print(f"Training Subset Size: {TRAIN_SUBSET_SIZE}")
    print(f"Epochs per fold: {EPOCHS}")

    # 2. Data Loading
    df_full_train = load_dataset_metadata("train")
    df_holdout_val = load_dataset_metadata("val")

    # Subsample training data for speed
    if len(df_full_train) > TRAIN_SUBSET_SIZE:
        print(
            f"Subsampling training data from {len(df_full_train)} to {TRAIN_SUBSET_SIZE}..."
        )
        # Stratified sample
        df_train = df_full_train.groupby("label", group_keys=False).apply(
            lambda x: x.sample(
                min(len(x), int(TRAIN_SUBSET_SIZE * len(x) / len(df_full_train))),
                random_state=Config.SEED,
            )
        )
        # Reset index
        df_train = df_train.reset_index(drop=True)
    else:
        df_train = df_full_train

    print(f"Train shape: {df_train.shape}")
    print(f"Holdout Val shape: {df_holdout_val.shape}")

    # 3. Cross-Validation Setup
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Store OOF predictions: key=model_name, value=array of shape (len(df_train),)
    oof_preds_storage = {arch: np.zeros(len(df_train)) for arch in Config.MODEL_ARCHS}

    # 4. Training Loop
    for arch in Config.MODEL_ARCHS:
        print(f"\n=== Training Architecture: {arch} ===")

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(df_train, df_train["label"])
        ):
            print(f"--- Fold {fold+1}/{Config.NUM_FOLDS} ---")

            # Prepare Datasets
            train_fold_df = df_train.iloc[train_idx]
            val_fold_df = df_train.iloc[val_idx]

            train_ds = PathologyDataset(
                train_fold_df, transforms=get_transforms("train")
            )
            val_ds = PathologyDataset(val_fold_df, transforms=get_transforms("valid"))

            train_loader = DataLoader(
                train_ds,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
                drop_last=True,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            # Model, Optimizer, Scheduler
            model = get_model(arch, pretrained=True)
            model.to(Config.DEVICE)

            optimizer = optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=EPOCHS, eta_min=Config.MIN_LR
            )

            # Training
            best_auc = 0
            model_save_path = os.path.join(
                Config.WORKING_DIR, f"{arch}_fold_{fold}.pth"
            )

            for epoch in range(EPOCHS):
                loss = train_one_epoch(
                    model, train_loader, optimizer, Config.DEVICE, epoch
                )
                auc, val_loss = valid_one_epoch(model, val_loader, Config.DEVICE, epoch)
                scheduler.step()

                if auc > best_auc:
                    best_auc = auc
                    save_checkpoint(model, optimizer, epoch, auc, model_save_path)

            print(f"Best AUC for Fold {fold}: {best_auc:.4f}")

            # Generate OOF predictions for this fold using TTA
            checkpoint = load_checkpoint(model, model_save_path)
            model.to(Config.DEVICE)

            # Create a loader for OOF inference
            oof_loader = DataLoader(
                val_ds,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            preds = tta_inference_fn(model, oof_loader, Config.DEVICE)
            oof_preds_storage[arch][val_idx] = preds

            # Clean up to free VRAM
            del model, optimizer, scheduler, train_loader, val_loader, oof_loader
            torch.cuda.empty_cache()

    # 5. Stacking
    print("\n=== Training Meta-Learner ===")
    # Train meta learner on the OOF predictions
    meta_auc = train_meta_learner(
        oof_preds_dict=oof_preds_storage,
        targets=df_train["label"].values,
        load_cached_data=False,
    )
    print(f"Stacking OOF AUC: {meta_auc:.4f}")

    # 6. Final Validation on Holdout Set
    print("\n=== Final Validation on Holdout Set ===")

    holdout_ds = PathologyDataset(df_holdout_val, transforms=get_transforms("valid"))
    holdout_loader = DataLoader(
        holdout_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    holdout_preds_dict = {}

    # Generate predictions for holdout set using all models
    for arch in Config.MODEL_ARCHS:
        arch_preds = []
        for fold in range(Config.NUM_FOLDS):
            model_path = os.path.join(Config.WORKING_DIR, f"{arch}_fold_{fold}.pth")
            model = get_model(arch, pretrained=False)
            load_checkpoint(model, model_path)
            model.to(Config.DEVICE)

            preds = tta_inference_fn(model, holdout_loader, Config.DEVICE)
            arch_preds.append(preds)

            del model
            torch.cuda.empty_cache()

        # Average across folds to get the single feature for this arch
        holdout_preds_dict[arch] = np.mean(arch_preds, axis=0)

    # Predict with Meta Learner
    final_val_preds = predict_with_meta_learner(holdout_preds_dict)

    # Calculate Metric
    final_auc = roc_auc_score(df_holdout_val["label"].values, final_val_preds)
    print(f"Final Validation Metric: {final_auc}")

    # 7. Failure Analysis
    analyze_failures(df_holdout_val, final_val_preds)

    # 8. Submission
    THRESHOLD = 0.9933607469455475
    if final_auc > THRESHOLD:
        print("\n=== Generating Submission ===")
        df_test = load_dataset_metadata("test")
        test_ds = PathologyDataset(df_test, transforms=get_transforms("test"))
        test_loader = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        test_preds_dict = {}
        for arch in Config.MODEL_ARCHS:
            arch_preds = []
            for fold in range(Config.NUM_FOLDS):
                model_path = os.path.join(Config.WORKING_DIR, f"{arch}_fold_{fold}.pth")
                model = get_model(arch, pretrained=False)
                load_checkpoint(model, model_path)
                model.to(Config.DEVICE)

                preds = tta_inference_fn(model, test_loader, Config.DEVICE)
                arch_preds.append(preds)

                del model
                torch.cuda.empty_cache()

            test_preds_dict[arch] = np.mean(arch_preds, axis=0)

        final_test_preds = predict_with_meta_learner(test_preds_dict)
        create_submission(df_test["id"].values, final_test_preds)
    else:
        print(
            f"\nSkipping submission. Validation AUC {final_auc} <= Threshold {THRESHOLD}"
        )


if __name__ == "__main__":
    run()
