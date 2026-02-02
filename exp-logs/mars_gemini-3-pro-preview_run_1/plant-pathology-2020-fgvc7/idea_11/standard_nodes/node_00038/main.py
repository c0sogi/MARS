import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
import cv2

# Add library path to ensure imports work correctly
sys.path.append("./library")

from library.config import Config
from library.utils import seed_everything, get_device, calculate_roc_auc
from library.dataset import AppleDataset, get_transforms
from library.model import AppleResNet34
from library.loss import WeightedCrossEntropyLoss, get_class_weights
from library.engine import train_one_epoch, valid_one_epoch, inference_fn


def main():
    # ==========================================
    # 1. Setup & Initialization
    # ==========================================
    seed_everything(Config.SEED)
    device = get_device()

    # Ensure output directories exist
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"Running {Config.IDEA_NAME} on {device}")

    # Load Metadata
    # train_metadata.csv: Used for Training (Phase 1 & 2)
    # val_metadata.csv: Used ONLY for Final Validation (Hold-out)
    # test_metadata.csv: Used for Submission
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # ==========================================
    # 2. Phase 1: Proxy Calibration (5-Fold CV)
    # ==========================================
    # Purpose: Find optimal epoch and determine TTA strategy
    print("\n==== Phase 1: Proxy Calibration (5-Fold CV) ====")

    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Storage for metrics
    fold_val_aucs = np.zeros((Config.N_FOLDS, Config.EPOCHS))
    fold_models = []  # Store (model, val_loader) for TTA check

    # Prepare stratification label
    if "stratify_label" in train_df.columns:
        y_strat = train_df["stratify_label"]
    else:
        y_strat = train_df[Config.CLASS_LABELS].idxmax(axis=1)

    # Calculate class weights once for the training set
    class_weights = get_class_weights(train_df, device=device)

    for fold, (train_idx, cv_val_idx) in enumerate(skf.split(train_df, y_strat)):
        # Create Fold DataFrames
        fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = train_df.iloc[cv_val_idx].reset_index(drop=True)

        # Datasets & Loaders
        train_dataset = AppleDataset(fold_train_df, transforms=get_transforms("train"))
        val_dataset = AppleDataset(fold_val_df, transforms=get_transforms("valid"))

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
        )

        # Initialize Model & Optimization
        model = AppleResNet34(pretrained=True).to(device)
        optimizer = optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler: Cosine Annealing synchronized with EPOCHS
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=Config.T_0, T_mult=Config.T_MULT, eta_min=Config.MIN_LR
        )

        criterion = WeightedCrossEntropyLoss(weights=class_weights, device=device)

        # Training Loop
        for epoch in range(Config.EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scheduler
            )
            val_loss, val_auc = valid_one_epoch(model, val_loader, criterion, device)
            fold_val_aucs[fold, epoch] = val_auc

        # Move model to CPU to save GPU memory, store for TTA check
        model.to("cpu")
        fold_models.append((model, val_loader))

    # --- Analyze Phase 1 Results ---

    # 1. Determine Optimal Epoch
    mean_auc_per_epoch = fold_val_aucs.mean(axis=0)
    best_epoch_idx = np.argmax(mean_auc_per_epoch)
    optimal_epochs = best_epoch_idx + 1
    print(
        f"Optimal Epoch determined via CV: {optimal_epochs} (AUC: {mean_auc_per_epoch[best_epoch_idx]:.5f})"
    )

    # 2. Determine TTA Strategy
    # We check TTA efficacy using the trained fold models
    print("Checking TTA efficacy...")
    auc_no_tta = []
    auc_tta = []

    for model, loader in fold_models:
        model.to(device)

        # Get ground truth
        all_targets = []
        for _, t in loader:
            all_targets.append(t.numpy())
        all_targets = np.concatenate(all_targets)

        # Inference without TTA
        preds_no_tta = inference_fn(model, loader, device, use_tta=False)
        auc_no_tta.append(calculate_roc_auc(all_targets, preds_no_tta))

        # Inference with TTA
        preds_tta = inference_fn(model, loader, device, use_tta=True)
        auc_tta.append(calculate_roc_auc(all_targets, preds_tta))

        model.to("cpu")

    mean_auc_no_tta = np.mean(auc_no_tta)
    mean_auc_tta = np.mean(auc_tta)

    use_tta = mean_auc_tta > mean_auc_no_tta
    print(
        f"TTA Check: No TTA={mean_auc_no_tta:.5f}, TTA={mean_auc_tta:.5f} -> Use TTA: {use_tta}"
    )

    # ==========================================
    # 3. Phase 2: Production Training
    # ==========================================
    # Purpose: Train ensemble on 100% of training data for optimal_epochs
    print("\n==== Phase 2: Production Training (Full-Data Seed Ensemble) ====")

    ensemble_models = []

    # Full Training Loader (using train_metadata.csv)
    full_train_dataset = AppleDataset(train_df, transforms=get_transforms("train"))
    full_train_loader = DataLoader(
        full_train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    for seed in Config.ENSEMBLE_SEEDS:
        # print(f"Training Seed Model: {seed}")
        seed_everything(seed)

        model = AppleResNet34(pretrained=True).to(device)
        optimizer = optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=Config.T_0, T_mult=Config.T_MULT, eta_min=Config.MIN_LR
        )
        criterion = WeightedCrossEntropyLoss(weights=class_weights, device=device)

        # Train for exactly optimal_epochs
        for epoch in range(optimal_epochs):
            _ = train_one_epoch(
                model, full_train_loader, optimizer, criterion, device, scheduler
            )

        model.eval()
        ensemble_models.append(model)

    # ==========================================
    # 4. Final Validation & Failure Analysis
    # ==========================================
    print("\n==== Final Validation on Hold-Out Set ====")

    val_dataset_holdout = AppleDataset(val_df, transforms=get_transforms("valid"))
    val_loader_holdout = DataLoader(
        val_dataset_holdout,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # 1. Ensemble Inference on Validation Set
    val_preds_list = []
    val_targets = []

    # Extract targets
    for _, t in val_loader_holdout:
        val_targets.append(t.numpy())
    val_targets = np.concatenate(val_targets)

    # Predict with each model
    for model in ensemble_models:
        preds = inference_fn(model, val_loader_holdout, device, use_tta=use_tta)
        val_preds_list.append(preds)

    # Average predictions
    avg_val_preds = np.mean(val_preds_list, axis=0)

    # Calculate Metric
    final_metric = calculate_roc_auc(val_targets, avg_val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 2. Failure Analysis
    print("\nPerforming Failure Analysis...")

    errors = []
    meta_stats = []

    # Iterate through validation set to calculate error and extract image stats
    for idx in range(len(val_df)):
        # Calculate Error Magnitude (1 - Probability of True Class)
        target_idx = val_targets[idx]
        prob_true = avg_val_preds[idx, target_idx]
        error = 1.0 - prob_true
        errors.append(error)

        # Extract Image Meta-Features
        row = val_df.iloc[idx]
        path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Read image
        img = cv2.imread(path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
            h, w, _ = img.shape

            meta_stats.append(
                {
                    "width": w,
                    "height": h,
                    "mean_intensity": img.mean(),
                    "mean_r": img[:, :, 0].mean(),
                    "mean_g": img[:, :, 1].mean(),
                    "mean_b": img[:, :, 2].mean(),
                }
            )
        else:
            # Fallback for missing images (should not happen per metadata check)
            meta_stats.append(
                {
                    "width": 0,
                    "height": 0,
                    "mean_intensity": 0,
                    "mean_r": 0,
                    "mean_g": 0,
                    "mean_b": 0,
                }
            )

    analysis_df = pd.DataFrame(meta_stats)
    analysis_df["error"] = errors

    # Calculate Correlation
    correlations = analysis_df.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # ==========================================
    # 5. Submission Generation
    # ==========================================
    threshold = 0.9871488489626378

    if final_metric > threshold:
        print("\nMetric check passed. Generating submission...")

        test_dataset = AppleDataset(test_df, transforms=get_transforms("test"))
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
        )

        test_preds_list = []
        for model in ensemble_models:
            preds = inference_fn(model, test_loader, device, use_tta=use_tta)
            test_preds_list.append(preds)

        avg_test_preds = np.mean(test_preds_list, axis=0)

        # Construct Submission DataFrame
        # Config.CLASS_LABELS = ["healthy", "multiple_diseases", "rust", "scab"]
        submission = pd.DataFrame(
            {
                "image_id": test_df["image_id"],
                "healthy": avg_test_preds[:, 0],
                "multiple_diseases": avg_test_preds[:, 1],
                "rust": avg_test_preds[:, 2],
                "scab": avg_test_preds[:, 3],
            }
        )

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(f"\nMetric {final_metric} <= {threshold}. Submission skipped.")


if __name__ == "__main__":
    main()
