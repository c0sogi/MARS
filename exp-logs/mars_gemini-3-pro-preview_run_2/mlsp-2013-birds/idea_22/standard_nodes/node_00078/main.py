import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from skmultilearn.model_selection import IterativeStratifiedKFold
from scipy.stats import pearsonr
import logging

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_score, get_logger
from library.models import BirdModel
from library.data import load_and_cache_images, BirdDataset, get_transforms
from library.loss import WeightedBCELoss, AnchorDistillationLoss
from library.sam import SAM
from library.engine import (
    train_one_epoch,
    valid_one_epoch,
    inference_fn,
    save_submission,
)

# Suppress verbose logs
logging.getLogger("transformers").setLevel(logging.ERROR)
import warnings

warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Create working directory for models
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Data Loading
    print("Loading metadata and images...")
    df_train_full = pd.read_csv(Config.TRAIN_CSV)
    df_val_holdout = pd.read_csv(Config.VAL_CSV)
    df_test = pd.read_csv(Config.TEST_CSV)

    # Load images into cache
    image_cache = load_and_cache_images(
        [df_train_full, df_val_holdout, df_test], load_cached_data=True
    )

    # 3. Create Folds
    # Using Iterative Stratified K-Fold for multi-label balance
    print(f"Generating {Config.N_FOLDS} folds...")
    X = df_train_full["rec_id"].values.reshape(-1, 1)
    label_cols = [c for c in df_train_full.columns if c.startswith("species_")]
    y = df_train_full[label_cols].values

    k_fold = IterativeStratifiedKFold(n_splits=Config.N_FOLDS, order=1)

    # Assign fold indices to dataframe
    df_train_full["fold"] = -1
    for fold_idx, (train_indices, val_indices) in enumerate(k_fold.split(X, y)):
        df_train_full.iloc[val_indices, df_train_full.columns.get_loc("fold")] = (
            fold_idx
        )

    # Dictionary to store OOF soft targets: rec_id -> prob_vector
    oof_soft_targets = {}

    # Store model paths for final ensemble
    model_paths = []

    # =========================================================================
    # STAGE 1: Train Anchors (ResNet18, EfficientNet-B0) & Generate OOFs
    # =========================================================================
    print("\n=== STAGE 1: Training Anchors ===")

    anchor_models = [Config.MODEL_RESNET, Config.MODEL_EFFICIENTNET]

    for fold in range(Config.N_FOLDS):
        print(f"-- Fold {fold} --")

        # Split Data
        train_df = df_train_full[df_train_full["fold"] != fold].reset_index(drop=True)
        valid_df = df_train_full[df_train_full["fold"] == fold].reset_index(drop=True)

        # Create Datasets/Loaders
        train_ds = BirdDataset(train_df, image_cache, get_transforms("train"))
        valid_ds = BirdDataset(valid_df, image_cache, get_transforms("valid"))

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Calculate positive weights for imbalance
        train_labels = train_df[label_cols].values
        pos_counts = np.sum(train_labels, axis=0)
        neg_counts = len(train_labels) - pos_counts
        # Clamp to avoid division by zero
        pos_weights = torch.tensor(
            neg_counts / (pos_counts + 1e-6), dtype=torch.float32
        ).to(device)

        fold_anchor_preds = []

        for model_name in anchor_models:
            print(f"Training {model_name}...")
            model = BirdModel(model_name, num_classes=Config.NUM_CLASSES).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=Config.LEARNING_RATE,
                steps_per_epoch=len(train_loader),
                epochs=Config.EPOCHS,
            )
            loss_fn = WeightedBCELoss(pos_weights=pos_weights)

            best_auc = 0
            best_model_path = os.path.join(
                Config.WORKING_DIR, f"{model_name}_fold{fold}.pth"
            )

            for epoch in range(Config.EPOCHS):
                train_loss = train_one_epoch(
                    model, train_loader, optimizer, scheduler, loss_fn, device, epoch
                )
                # Quick validation
                if (epoch + 1) % 5 == 0 or epoch == Config.EPOCHS - 1:
                    val_loss, val_auc = valid_one_epoch(
                        model, valid_loader, loss_fn, device
                    )
                    if val_auc > best_auc:
                        best_auc = val_auc
                        torch.save(model.state_dict(), best_model_path)

            # Load best and predict OOF
            model.load_state_dict(torch.load(best_model_path))
            model.eval()
            preds = []
            rec_ids = []
            with torch.no_grad():
                for imgs, _, _, ids in valid_loader:
                    imgs = imgs.to(device)
                    logits = model(imgs)
                    preds.append(torch.sigmoid(logits).cpu().numpy())
                    rec_ids.append(ids.numpy())

            preds = np.concatenate(preds)
            rec_ids = np.concatenate(rec_ids)

            # Organize preds by rec_id
            pred_map = {rid: p for rid, p in zip(rec_ids, preds)}
            fold_anchor_preds.append(pred_map)
            model_paths.append(best_model_path)

        # Average Anchor Predictions for Soft Targets
        for rid in valid_df["rec_id"].values:
            # Average predictions from ResNet and EfficientNet
            avg_pred = np.mean([f_preds[rid] for f_preds in fold_anchor_preds], axis=0)
            oof_soft_targets[rid] = avg_pred

    # =========================================================================
    # STAGE 2: Train Student (DenseNet121) with SAM & Distillation
    # =========================================================================
    print("\n=== STAGE 2: Training Student (DenseNet121 + SAM) ===")

    student_model_name = Config.MODEL_DENSENET

    for fold in range(Config.N_FOLDS):
        print(f"-- Fold {fold} --")

        train_df = df_train_full[df_train_full["fold"] != fold].reset_index(drop=True)
        valid_df = df_train_full[df_train_full["fold"] == fold].reset_index(drop=True)

        # Pass soft targets to dataset
        train_ds = BirdDataset(
            train_df,
            image_cache,
            get_transforms("train"),
            soft_targets=oof_soft_targets,
        )
        valid_ds = BirdDataset(
            valid_df, image_cache, get_transforms("valid")
        )  # No soft targets needed for val

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Weights
        train_labels = train_df[label_cols].values
        pos_counts = np.sum(train_labels, axis=0)
        neg_counts = len(train_labels) - pos_counts
        pos_weights = torch.tensor(
            neg_counts / (pos_counts + 1e-6), dtype=torch.float32
        ).to(device)

        # Model & SAM Optimizer
        model = BirdModel(student_model_name, num_classes=Config.NUM_CLASSES).to(device)
        base_optimizer = optim.AdamW
        optimizer = SAM(
            model.parameters(),
            base_optimizer,
            rho=Config.SAM_RHO,
            lr=Config.LEARNING_RATE,
        )

        # Scheduler (Note: SAM steps per epoch is same as loader length)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer.base_optimizer,
            max_lr=Config.LEARNING_RATE,
            steps_per_epoch=len(train_loader),
            epochs=Config.EPOCHS,
        )

        # Distillation Loss
        loss_fn = AnchorDistillationLoss(
            pos_weights=pos_weights, distillation_lambda=Config.DISTILLATION_LAMBDA
        )

        best_auc = 0
        best_model_path = os.path.join(
            Config.WORKING_DIR, f"{student_model_name}_fold{fold}.pth"
        )

        for epoch in range(Config.EPOCHS):
            # Train with SAM
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, loss_fn, device, epoch
            )

            if (epoch + 1) % 5 == 0 or epoch == Config.EPOCHS - 1:
                # Validate (using hard loss only for metric)
                val_loss, val_auc = valid_one_epoch(
                    model, valid_loader, WeightedBCELoss(), device
                )
                if val_auc > best_auc:
                    best_auc = val_auc
                    torch.save(model.state_dict(), best_model_path)

        model_paths.append(best_model_path)

    # =========================================================================
    # FINAL EVALUATION: Ensemble Inference on Hold-out Validation Set
    # =========================================================================
    print("\n=== Final Evaluation on Hold-out Set ===")

    val_holdout_ds = BirdDataset(df_val_holdout, image_cache, get_transforms("valid"))
    val_holdout_loader = DataLoader(
        val_holdout_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    ensemble_preds = []

    # Iterate over all 15 models (5 folds * 3 architectures)
    for path in model_paths:
        # Determine architecture from filename
        if "resnet18" in path:
            arch = "resnet18"
        elif "efficientnet" in path:
            arch = "efficientnet_b0"
        elif "densenet" in path:
            arch = "densenet121"
        else:
            continue

        model = BirdModel(arch, num_classes=Config.NUM_CLASSES).to(device)
        model.load_state_dict(torch.load(path))

        # Inference with TTA
        preds, ids = inference_fn(model, val_holdout_loader, device)
        ensemble_preds.append(preds)

    # Average predictions
    avg_val_preds = np.mean(ensemble_preds, axis=0)

    # Calculate Metric
    val_targets = df_val_holdout[label_cols].values
    final_metric = get_score(val_targets, avg_val_preds)

    print(f"Final Validation Metric: {final_metric}")

    # =========================================================================
    # FAILURE ANALYSIS
    # =========================================================================
    print("\n=== Failure Analysis ===")

    # Calculate per-sample error (Mean BCE across classes)
    # Clip preds for stability
    y_pred_clipped = np.clip(avg_val_preds, 1e-7, 1 - 1e-7)
    # BCE = -(y log(p) + (1-y) log(1-p))
    bce_matrix = -(
        val_targets * np.log(y_pred_clipped)
        + (1 - val_targets) * np.log(1 - y_pred_clipped)
    )
    sample_errors = np.mean(bce_matrix, axis=1)

    # Extract features from cached images (Mean, Std)
    img_means = []
    img_stds = []
    for rid in df_val_holdout["rec_id"].values:
        img = image_cache[rid]
        img_means.append(np.mean(img))
        img_stds.append(np.std(img))

    # Correlation
    corr_mean, _ = pearsonr(sample_errors, img_means)
    corr_std, _ = pearsonr(sample_errors, img_stds)

    print(f"Correlation (Error vs Image Mean): {corr_mean:.4f}")
    print(f"Correlation (Error vs Image Std): {corr_std:.4f}")

    # =========================================================================
    # SUBMISSION
    # =========================================================================
    threshold = 0.9167709334579945

    if final_metric > threshold:
        print("\nMetric exceeds threshold. Generating submission...")

        test_ds = BirdDataset(
            df_test, image_cache, get_transforms("valid"), is_test=True
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        test_ensemble_preds = []
        test_rec_ids = None

        for path in model_paths:
            if "resnet18" in path:
                arch = "resnet18"
            elif "efficientnet" in path:
                arch = "efficientnet_b0"
            elif "densenet" in path:
                arch = "densenet121"

            model = BirdModel(arch, num_classes=Config.NUM_CLASSES).to(device)
            model.load_state_dict(torch.load(path))

            preds, ids = inference_fn(model, test_loader, device)
            test_ensemble_preds.append(preds)
            test_rec_ids = ids  # IDs are same for all

        avg_test_preds = np.mean(test_ensemble_preds, axis=0)

        sub_path = os.path.join("submission", "submission.csv")
        save_submission(avg_test_preds, test_rec_ids, sub_path)
    else:
        print(
            f"\nMetric {final_metric} did not exceed threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
