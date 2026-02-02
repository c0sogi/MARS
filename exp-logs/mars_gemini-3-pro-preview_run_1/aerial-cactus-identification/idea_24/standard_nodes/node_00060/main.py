import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr

# Import from library
from library.config import Config
from library.utils import set_seed, calculate_roc_auc
from library.data import load_and_process_data, CactusDataset, get_transforms
from library.model import MultiHeadRepVGG
from library.engine import train_fold_swa, predict_tta


def run_pipeline():
    # 1. Setup
    Config.setup()
    device = torch.device(Config.DEVICE)
    set_seed(Config.SEED)

    print(f"Device: {device}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Folds: {Config.N_FOLDS}")

    # 2. Data Loading
    print("\n[Data] Loading datasets...")

    # Load Train Data (from train_metadata.csv) - used for 5-Fold CV
    train_imgs, train_targets = load_and_process_data(
        Config.TRAIN_METADATA_PATH,
        "train",
        load_cached_data=Config.USE_CACHE,
        is_test=False,
        debug=Config.DEBUG,
    )

    # Load Hold-out Validation Data (from val_metadata.csv) - used for Final Metric
    val_imgs, val_targets = load_and_process_data(
        Config.VAL_METADATA_PATH,
        "val",
        load_cached_data=Config.USE_CACHE,
        is_test=False,
        debug=Config.DEBUG,
    )

    # Load Test Data (from test_metadata.csv)
    test_imgs, test_ids = load_and_process_data(
        Config.TEST_METADATA_PATH,
        "test",
        load_cached_data=Config.USE_CACHE,
        is_test=True,
        debug=Config.DEBUG,
    )

    print(f"Train set shape: {train_imgs.shape}")
    print(f"Hold-out Val set shape: {val_imgs.shape}")
    print(f"Test set shape: {test_imgs.shape}")

    # 3. 5-Fold Stratified Cross-Validation Training
    # We split the 'train_imgs' into 5 folds.
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    fold_models = []

    for fold_idx, (train_indices, val_indices) in enumerate(
        skf.split(train_imgs, train_targets)
    ):
        print(f"\n[Fold {fold_idx}] Preparing data...")

        # Split Data
        X_fold_train = train_imgs[train_indices]
        y_fold_train = train_targets[train_indices]
        X_fold_val = train_imgs[val_indices]
        y_fold_val = train_targets[val_indices]

        # Create Datasets & Loaders
        train_ds = CactusDataset(
            X_fold_train, y_fold_train, transform=get_transforms("train")
        )
        val_ds = CactusDataset(X_fold_val, y_fold_val, transform=get_transforms("val"))

        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model
        model = MultiHeadRepVGG(deploy=False).to(device)

        # Train with SWA
        trained_model = train_fold_swa(
            model, train_loader, val_loader, fold_idx, device
        )

        # Move to CPU to conserve GPU memory
        trained_model.cpu()
        fold_models.append(trained_model)

        # Cleanup
        del model, train_loader, val_loader, X_fold_train, X_fold_val
        torch.cuda.empty_cache()

    # 4. Final Evaluation on Hold-out Validation Set
    print("\n[Validation] Evaluating Ensemble on Hold-out Set...")

    val_ds_holdout = CactusDataset(
        val_imgs, val_targets, transform=get_transforms("val")
    )
    val_loader_holdout = torch.utils.data.DataLoader(
        val_ds_holdout,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    ensemble_preds = np.zeros(len(val_targets))

    for i, model in enumerate(fold_models):
        print(f"  Inference with Fold {i} Model...")
        model.to(device)
        model.switch_to_deploy()  # Fuse blocks for inference speed
        model.eval()

        preds = predict_tta(model, val_loader_holdout, device)
        ensemble_preds += preds

        model.cpu()  # Move back to CPU
        torch.cuda.empty_cache()

    # Average predictions
    ensemble_preds /= Config.N_FOLDS

    # Compute Metric
    final_metric = calculate_roc_auc(val_targets, ensemble_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n[Analysis] Performing Failure Analysis...")

    # Calculate absolute error
    errors = np.abs(val_targets - ensemble_preds)

    # Compute Meta-features for Hold-out Set
    # Intensity (Mean of pixels)
    # val_imgs is (N, 3, 32, 32)
    mean_intensities = val_imgs.mean(axis=(1, 2, 3))

    # Contrast (Std of pixels)
    contrasts = val_imgs.std(axis=(1, 2, 3))

    # File Sizes
    # We need to reload metadata to get paths
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    file_sizes = []
    for rel_path in val_df["file_path"]:
        full_path = os.path.join(Config.INPUT_DIR, rel_path)
        if os.path.exists(full_path):
            file_sizes.append(os.path.getsize(full_path))
        else:
            file_sizes.append(0)
    file_sizes = np.array(file_sizes)

    # Correlations
    # Handle edge case where std is 0 (constant images) though unlikely
    corr_int = (
        pearsonr(errors, mean_intensities)[0] if np.std(mean_intensities) > 0 else 0
    )
    corr_cont = pearsonr(errors, contrasts)[0] if np.std(contrasts) > 0 else 0
    corr_fs = pearsonr(errors, file_sizes)[0] if np.std(file_sizes) > 0 else 0

    print(f"Correlation (Error vs Mean Intensity): {corr_int:.4f}")
    print(f"Correlation (Error vs Contrast):       {corr_cont:.4f}")
    print(f"Correlation (Error vs File Size):      {corr_fs:.4f}")

    # 6. Submission
    # Proceed if metric is valid (using 0.5 as baseline for random guessing)
    if final_metric > 0.5:
        print("\n[Submission] Generating Test Predictions...")

        test_ds = CactusDataset(
            test_imgs, test_ids, transform=get_transforms("test"), is_test=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        test_ensemble_preds = np.zeros(len(test_ids))

        for i, model in enumerate(fold_models):
            # Model is already in deploy mode from validation step
            model.to(device)
            model.eval()

            preds = predict_tta(model, test_loader, device)
            test_ensemble_preds += preds

            model.cpu()
            torch.cuda.empty_cache()

        test_ensemble_preds /= Config.N_FOLDS

        # Create Submission DataFrame
        sub_df = pd.DataFrame({"id": test_ids, "has_cactus": test_ensemble_preds})

        sub_df.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
    else:
        print(f"Validation metric {final_metric} is too low. Skipping submission.")


if __name__ == "__main__":
    run_pipeline()
