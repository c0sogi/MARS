import sys
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
import cv2

# Import from library
from library.config import Config
from library.utils import set_seed, MetricMonitor, save_checkpoint
from library.dataset import load_data, CactusDataset, get_transforms
from library.models import get_model
from library.train_eval import train_one_epoch, validate, predict_tta
from library.ensemble import EnsembleStacker, save_submission


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Adjust Config for the "Fast Baseline" requirement while ensuring high performance.
    # Extend to 35 Epochs to ensure convergence with Mixup (Cite {solution_lesson_node_00016})
    # while maintaining the 3-model diversity (Cite {solution_lesson_node_00018}).
    Config.EPOCHS = 35
    Config.NUM_FOLDS = 5

    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Running on device: {device}")
    print(f"Configuration: {Config.EPOCHS} Epochs, {Config.NUM_FOLDS} Folds")

    # ==========================================
    # 2. Load Data
    # ==========================================
    print("\nLoading Data...")
    # Train data (used for Cross-Validation)
    train_imgs, train_lbls, train_ids = load_data(
        Config.TRAIN_METADATA_PATH, Config.INPUT_DIR, "train", load_cached_data=True
    )
    # Holdout Validation data (used for Final Metric & Failure Analysis)
    val_imgs, val_lbls, val_ids = load_data(
        Config.VAL_METADATA_PATH, Config.INPUT_DIR, "val", load_cached_data=True
    )
    # Test data (used for Submission)
    test_imgs, test_lbls, test_ids = load_data(
        Config.TEST_METADATA_PATH, Config.INPUT_DIR, "test", load_cached_data=True
    )

    # ==========================================
    # 3. Training Loop (Models x Folds)
    # ==========================================
    # Storage for ensemble inputs
    model_oof_preds = {}  # {model_name: {id: prob}}
    model_val_preds = {}  # {model_name: {id: prob}} (Averaged over folds)
    model_test_preds = {}  # {model_name: {id: prob}} (Averaged over folds)

    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    for model_name in Config.MODEL_ARCHS:
        print(f"\n=== Processing Architecture: {model_name} ===")

        # Storage for this architecture across folds
        this_arch_oof = {}
        this_arch_val_accum = {id_: 0.0 for id_ in val_ids}
        this_arch_test_accum = {id_: 0.0 for id_ in test_ids}

        # Iterate Folds
        for fold, (train_idx, cv_val_idx) in enumerate(
            skf.split(train_imgs, train_lbls)
        ):
            print(f"  Fold {fold+1}/{Config.NUM_FOLDS}")

            # Split Data
            X_train, X_cv_val = train_imgs[train_idx], train_imgs[cv_val_idx]
            y_train, y_cv_val = train_lbls[train_idx], train_lbls[cv_val_idx]
            ids_train, ids_cv_val = train_ids[train_idx], train_ids[cv_val_idx]

            # Create Datasets
            train_ds = CactusDataset(
                X_train, y_train, ids_train, transform=get_transforms("train")
            )
            cv_val_ds = CactusDataset(
                X_cv_val, y_cv_val, ids_cv_val, transform=get_transforms("val")
            )

            # Create Loaders
            train_loader = DataLoader(
                train_ds,
                batch_size=Config.BATCH_SIZE,
                shuffle=True,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )
            val_loader = DataLoader(
                cv_val_ds,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            # Initialize Model
            model = get_model(
                model_name, num_classes=Config.NUM_CLASSES, pretrained=True
            )
            model = model.to(device)

            # Optimizer & Scheduler
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=Config.EPOCHS
            )

            # Training
            best_auc = 0.0
            best_model_state = None

            for epoch in range(Config.EPOCHS):
                _ = train_one_epoch(model, train_loader, optimizer, device, epoch)
                _, val_auc = validate(model, val_loader, device)
                scheduler.step()

                if val_auc > best_auc:
                    best_auc = val_auc
                    best_model_state = model.state_dict()

            # Load best weights
            if best_model_state is not None:
                model.load_state_dict(best_model_state)

            # 1. Generate OOF Predictions (using TTA for consistency)
            # We need a loader that yields IDs, so we use predict_tta logic
            cv_val_loader_tta = DataLoader(
                cv_val_ds,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
            )
            fold_oof_preds = predict_tta(model, cv_val_loader_tta, device)
            this_arch_oof.update(fold_oof_preds)

            # 2. Predict on Holdout Validation Set (Accumulate)
            holdout_ds = CactusDataset(
                val_imgs, val_lbls, val_ids, transform=get_transforms("val")
            )
            holdout_loader = DataLoader(
                holdout_ds,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
            )
            fold_holdout_preds = predict_tta(model, holdout_loader, device)

            for id_, prob in fold_holdout_preds.items():
                this_arch_val_accum[id_] += prob

            # 3. Predict on Test Set (Accumulate)
            test_ds = CactusDataset(
                test_imgs, test_lbls, test_ids, transform=get_transforms("test")
            )
            test_loader = DataLoader(
                test_ds,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
            )
            fold_test_preds = predict_tta(model, test_loader, device)

            for id_, prob in fold_test_preds.items():
                this_arch_test_accum[id_] += prob

            # Cleanup
            del model, optimizer, scheduler, train_loader, val_loader
            torch.cuda.empty_cache()

        # Average predictions over folds
        this_arch_val_avg = {
            k: v / Config.NUM_FOLDS for k, v in this_arch_val_accum.items()
        }
        this_arch_test_avg = {
            k: v / Config.NUM_FOLDS for k, v in this_arch_test_accum.items()
        }

        # Store in global dicts
        model_oof_preds[model_name] = this_arch_oof
        model_val_preds[model_name] = this_arch_val_avg
        model_test_preds[model_name] = this_arch_test_avg

    # ==========================================
    # 4. Stacked Ensemble
    # ==========================================
    print("\n=== Training Ensemble Stacker ===")
    stacker = EnsembleStacker()

    # Map Train IDs to labels for OOF training
    train_gt_dict = dict(zip(train_ids, train_lbls))

    # Fit Stacker
    stacker.fit_meta_learner(model_oof_preds, train_gt_dict)

    # ==========================================
    # 5. Final Evaluation
    # ==========================================
    print("\n=== Final Evaluation on Holdout Validation Set ===")
    # Predict using stacker on holdout set
    val_ensemble_preds = stacker.predict_ensemble(model_val_preds)

    # Align predictions with ground truth
    val_gt_dict = dict(zip(val_ids, val_lbls))

    y_true_sorted = []
    y_pred_sorted = []

    # Ensure order is consistent
    sorted_ids = sorted(val_ensemble_preds.keys())
    for id_ in sorted_ids:
        y_true_sorted.append(val_gt_dict[id_])
        y_pred_sorted.append(val_ensemble_preds[id_])

    final_metric = roc_auc_score(y_true_sorted, y_pred_sorted)

    # Print Metric with full precision
    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 6. Failure Analysis
    # ==========================================
    print("\n=== Failure Analysis ===")
    errors = np.abs(np.array(y_true_sorted) - np.array(y_pred_sorted))

    # Map ID to index in val_imgs to retrieve image data
    id_to_idx = {id_: i for i, id_ in enumerate(val_ids)}

    brightness_list = []
    contrast_list = []
    error_list = []

    for i, id_ in enumerate(sorted_ids):
        img_idx = id_to_idx[id_]
        img = val_imgs[img_idx]

        # Calculate simple stats
        brightness_list.append(img.mean())
        contrast_list.append(img.std())
        error_list.append(errors[i])

    # Calculate Correlations
    corr_bright, _ = pearsonr(error_list, brightness_list)
    corr_contrast, _ = pearsonr(error_list, contrast_list)

    print(f"Correlation between Error and Brightness: {corr_bright:.4f}")
    print(f"Correlation between Error and Contrast: {corr_contrast:.4f}")

    # ==========================================
    # 7. Submission
    # ==========================================
    THRESHOLD = 0.9999953560392056

    if final_metric > THRESHOLD:
        print(f"\nMetric {final_metric} > {THRESHOLD}. Generating submission...")
        test_ensemble_preds = stacker.predict_ensemble(model_test_preds)
        save_submission(test_ensemble_preds, Config.SUBMISSION_PATH)
    else:
        print(f"\nMetric {final_metric} <= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
