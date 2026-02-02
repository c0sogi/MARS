# Outline:
# 1. Initialize environment and override Config for fast execution (12 epochs, SWA start 8).
# 2. Load training and test data using data_manager.
# 3. Normalize file size features.
# 4. Perform 5-Fold Stratified Cross-Validation.
# 5. Inside CV loop:
#    a. Instantiate RepVGG, ResNet, NeXt models.
#    b. Train each using training_engine (train_one_epoch, validate).
#    c. Save best checkpoints.
#    d. Generate OOF predictions and Test predictions using inference_engine (with TTA).
# 6. Aggregate OOF predictions and train Stacking Meta-Learner.
# 7. Calculate and print Final Validation Metric (ROC AUC).
# 8. Perform Failure Analysis (Correlation of Error vs FileSize/Mean/Std).
# 9. Aggregate Test predictions across folds.
# 10. Generate final ensemble predictions and save submission.csv.

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.utils import (
    seed_everything,
    get_logger,
    save_checkpoint,
    calculate_roc_auc,
)
from library.data_manager import (
    load_all_train_data,
    load_test_data,
    get_file_size_stats,
    normalize_file_sizes,
    get_transforms,
    CactusDataset,
)
from library.models import RepVGG_FiLM, ResNet_FiLM, NeXt_FiLM
from library.training_engine import train_one_epoch, validate, SWAHandler
from library.inference_engine import predict_loader
from library.stacking import fit_meta_learner, predict_ensemble, save_submission


def main():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration Override
    # -------------------------------------------------------------------------
    seed_everything(Config.SEED)

    # Override Config for fast baseline execution
    Config.EPOCHS = 12  # Reduced from 35 to ensure < 2 hours
    Config.SWA_START_EPOCH = 8  # Start SWA earlier
    Config.FOLDS = 5  # Keep 5 folds for stability

    print("=" * 40)
    print(" STARTING PIPELINE")
    print(f" Epochs: {Config.EPOCHS}")
    print(f" Folds: {Config.FOLDS}")
    print("=" * 40)

    # Ensure directories exist
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Data Loading & Preprocessing
    # -------------------------------------------------------------------------
    print("\n[Data] Loading Datasets...")
    train_imgs, train_labels, train_ids, train_fsizes = load_all_train_data(
        load_cached_data=True
    )
    test_imgs, _, test_ids, test_fsizes = load_test_data(load_cached_data=True)

    # Normalize File Sizes based on Training Statistics
    fs_mean, fs_std = get_file_size_stats(train_fsizes)
    train_fsizes_norm = normalize_file_sizes(train_fsizes, fs_mean, fs_std)
    test_fsizes_norm = normalize_file_sizes(test_fsizes, fs_mean, fs_std)

    print(f"[Data] Train shape: {train_imgs.shape}, Test shape: {test_imgs.shape}")

    # -------------------------------------------------------------------------
    # 3. Cross-Validation Loop
    # -------------------------------------------------------------------------
    skf = StratifiedKFold(n_splits=Config.FOLDS, shuffle=True, random_state=Config.SEED)

    model_names = Config.MODELS
    # Arrays to store Out-Of-Fold predictions
    oof_preds_dict = {m: np.zeros(len(train_labels)) for m in model_names}
    # Lists to store Test predictions from each fold
    test_preds_dict = {m: [] for m in model_names}

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_imgs, train_labels)):
        print(f"\n{'='*20} Fold {fold+1}/{Config.FOLDS} {'='*20}")

        # Prepare Fold Data
        X_train, y_train = train_imgs[train_idx], train_labels[train_idx]
        s_train = train_fsizes_norm[train_idx]
        ids_train = train_ids[train_idx]

        X_val, y_val = train_imgs[val_idx], train_labels[val_idx]
        s_val = train_fsizes_norm[val_idx]
        ids_val = train_ids[val_idx]

        # Create Datasets
        train_dataset = CactusDataset(
            X_train, y_train, s_train, ids_train, transform=get_transforms("train")
        )
        val_dataset = CactusDataset(
            X_val, y_val, s_val, ids_val, transform=get_transforms("val")
        )
        # Test Dataset (Full)
        test_dataset = CactusDataset(
            test_imgs,
            np.zeros(len(test_imgs)),
            test_fsizes_norm,
            test_ids,
            transform=get_transforms("test"),
        )

        # Create Loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Train each model architecture
        for model_name in model_names:
            print(f"\n[Fold {fold+1}] Training {model_name}...")

            # Instantiate Model
            if model_name == "RepVGG_FiLM":
                model = RepVGG_FiLM(num_classes=1)
            elif model_name == "ResNet_FiLM":
                model = ResNet_FiLM(num_classes=1)
            elif model_name == "NeXt_FiLM":
                model = NeXt_FiLM(num_classes=1)
            else:
                continue

            model = model.to(Config.DEVICE)

            # Setup Training Components
            optimizer = optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=Config.EPOCHS
            )
            criterion = nn.BCEWithLogitsLoss()
            swa_handler = SWAHandler(model, optimizer, Config.DEVICE)

            best_auc = 0.0
            best_model_path = os.path.join(
                Config.CHECKPOINT_DIR, f"{model_name}_fold{fold}_best.pth"
            )

            # Training Epochs
            for epoch in range(Config.EPOCHS):
                train_loss = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    criterion,
                    Config.DEVICE,
                    epoch,
                    scheduler,
                )

                # Check SWA
                swa_handler.step(epoch)

                # Validate
                val_loss, val_auc = validate(
                    model, val_loader, criterion, Config.DEVICE
                )

                if val_auc > best_auc:
                    best_auc = val_auc
                    save_checkpoint(
                        {"state_dict": model.state_dict(), "auc": val_auc},
                        True,
                        best_model_path,
                    )

            # Load Best Model for Inference
            # (Skipping SWA inference for simplicity/speed in baseline, relying on best val score)
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint["state_dict"])

            # Generate OOF Predictions
            oof_probs, _ = predict_loader(model, val_loader, Config.DEVICE)
            oof_preds_dict[model_name][val_idx] = oof_probs

            # Generate Test Predictions
            test_probs, _ = predict_loader(model, test_loader, Config.DEVICE)
            test_preds_dict[model_name].append(test_probs)

            # Cleanup
            del model, optimizer, swa_handler, criterion
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 4. Stacking & Evaluation
    # -------------------------------------------------------------------------
    print("\n[Stacking] Training Meta-Learner on OOF predictions...")
    meta_learner = fit_meta_learner(oof_preds_dict, train_labels)

    # Evaluate Stacking Performance
    oof_meta_preds = meta_learner.predict(oof_preds_dict)
    final_auc = calculate_roc_auc(train_labels, oof_meta_preds)

    print(f"Final Validation Metric: {final_auc:.10f}")

    # -------------------------------------------------------------------------
    # 5. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n[Analysis] Performing Failure Analysis...")
    errors = np.abs(train_labels - oof_meta_preds)

    # Calculate image statistics for correlation
    print("  Computing image statistics...")
    img_means = []
    img_stds = []

    # Process in chunks
    chunk_size = 2000
    for i in range(0, len(train_imgs), chunk_size):
        chunk = train_imgs[i : i + chunk_size].astype(np.float32) / 255.0
        # Mean/Std over (H, W, C)
        img_means.extend(chunk.mean(axis=(1, 2, 3)))
        img_stds.extend(chunk.std(axis=(1, 2, 3)))

    img_means = np.array(img_means)
    img_stds = np.array(img_stds)

    # Compute Correlations
    corr_fsize, _ = pearsonr(errors, train_fsizes_norm)
    corr_mean, _ = pearsonr(errors, img_means)
    corr_std, _ = pearsonr(errors, img_stds)

    print("  Correlation between Error Magnitude and Input Features:")
    print(f"    File Size:            {corr_fsize:.4f}")
    print(f"    Image Mean Intensity: {corr_mean:.4f}")
    print(f"    Image Contrast (Std): {corr_std:.4f}")

    # -------------------------------------------------------------------------
    # 6. Submission
    # -------------------------------------------------------------------------
    # Average test predictions across folds
    final_test_preds_dict = {}
    for m in model_names:
        # stack list of arrays -> (Folds, N_test) -> mean -> (N_test,)
        final_test_preds_dict[m] = np.mean(np.stack(test_preds_dict[m]), axis=0)

    # Meta-learner prediction
    final_submission_probs = predict_ensemble(meta_learner, final_test_preds_dict)

    # Save Submission
    save_submission(test_ids, final_submission_probs)


if __name__ == "__main__":
    main()
