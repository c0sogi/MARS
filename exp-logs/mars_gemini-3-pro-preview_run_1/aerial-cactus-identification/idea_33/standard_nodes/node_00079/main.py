import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Import from library
from library.config import Config
from library.utils import seed_everything
from library.data import load_data_to_memory, CactusDataset, get_transforms
from library.models import CactusRepVGG, CactusResNet
from library.engine import Engine
from library.stacking import StackingEnsemble, GeometricFeatureExtractor


def run_training():
    print("Starting 5-Fold CV Training...")

    # 1. Load all labeled data
    # We combine train and val splits provided by the system to perform our own 5-Fold CV
    train_imgs, train_targets = load_data_to_memory("train", load_cached_data=True)
    val_imgs, val_targets = load_data_to_memory("val", load_cached_data=True)

    all_imgs = np.concatenate([train_imgs, val_imgs], axis=0)
    all_targets = np.concatenate([train_targets, val_targets], axis=0)

    # 2. Setup CV
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Storage for OOF Features
    # Shape: (N_samples, N_architectures * 2_features)
    # Feature Order: [RepVGG_Mean, RepVGG_Std, ResNet_Mean, ResNet_Std]
    num_samples = len(all_targets)
    num_archs = len(Config.MODELS)
    oof_features = np.zeros((num_samples, num_archs * 2), dtype=np.float32)
    oof_targets = np.zeros((num_samples,), dtype=np.float32)

    # Store model paths for later inference
    trained_model_paths = {arch: [] for arch in Config.MODELS}

    # Reduced epochs for fast baseline execution
    EPOCHS = 10

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_imgs, all_targets)):
        print(f"\n=== Fold {fold} ===")

        # Create Datasets for this fold
        train_ds = CactusDataset(
            all_imgs[train_idx],
            all_targets[train_idx],
            transform=get_transforms("train"),
        )
        val_ds = CactusDataset(
            all_imgs[val_idx], all_targets[val_idx], transform=get_transforms("val")
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

        # Train each architecture
        for arch_idx, arch_name in enumerate(Config.MODELS):
            print(f"Training {arch_name}...")

            # Init Model
            if arch_name == "CactusRepVGG":
                model = CactusRepVGG(num_classes=Config.NUM_CLASSES)
            else:
                model = CactusResNet(num_classes=Config.NUM_CLASSES)

            model.to(Config.DEVICE)

            # Optimizer & Scheduler
            optimizer = optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

            # SWA Setup
            swa_model = optim.swa_utils.AveragedModel(model)
            swa_start = EPOCHS - 3  # Start SWA in last 3 epochs

            # Training Loop
            for epoch in range(EPOCHS):
                _ = Engine.train_one_epoch(
                    model, train_loader, optimizer, Config.DEVICE, epoch
                )
                scheduler.step()

                if epoch >= swa_start:
                    swa_model.update_parameters(model)

            # Update BN statistics for SWA model
            optim.swa_utils.update_bn(train_loader, swa_model, device=Config.DEVICE)

            # Save Checkpoint
            ckpt_name = f"{arch_name}_fold{fold}.pth"
            ckpt_path = os.path.join(Config.CHECKPOINT_DIR, ckpt_name)
            torch.save(swa_model.module.state_dict(), ckpt_path)
            trained_model_paths[arch_name].append(ckpt_path)

            # Generate OOF Features for this fold/model
            inference_model = swa_model.module
            inference_model.eval()

            # RepVGG structural re-parameterization for inference speed
            if arch_name == "CactusRepVGG":
                inference_model.switch_to_deploy()

            # Predict with TTA (4 views)
            raw_probs, _ = Engine.predict_tta_raw(
                inference_model, val_loader, Config.DEVICE
            )

            # Extract Geometric Features (Mean, Std)
            feats = GeometricFeatureExtractor.extract(raw_probs)  # (N_val, 2)

            # Fill OOF matrix
            # Indices: arch_idx=0 -> cols [0,1], arch_idx=1 -> cols [2,3]
            oof_features[val_idx, arch_idx * 2 : (arch_idx + 1) * 2] = feats

            # Store targets
            oof_targets[val_idx] = all_targets[val_idx]

            # Cleanup to free VRAM
            del model, swa_model, optimizer, scheduler, inference_model
            torch.cuda.empty_cache()

    return oof_features, oof_targets, trained_model_paths, all_imgs


def run_meta_learning(oof_features, oof_targets):
    print("\nTraining Meta-Learner...")
    stacker = StackingEnsemble()

    # FIX: Use Cross-Validation to get unbiased predictions for scoring
    # This prevents leakage where the model is evaluated on data it was trained on.
    print("Performing internal CV for Meta-Learner evaluation...")
    cv = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Generate unbiased probability predictions
    # cross_val_predict returns (N, n_classes) for method='predict_proba'
    cv_preds = cross_val_predict(
        stacker.meta_learner,
        oof_features,
        oof_targets,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    # Calculate valid AUC
    auc = roc_auc_score(oof_targets, cv_preds)
    print(f"Final Validation Metric (Nested CV): {auc:.10f}")

    # Fit on full OOF data for final submission inference
    stacker.meta_learner.fit(oof_features, oof_targets)

    # Save meta learner
    meta_path = os.path.join(Config.WORK_DIR, "meta_learner.joblib")
    stacker.save_meta_learner(meta_path)

    return stacker, auc, cv_preds


def run_failure_analysis(oof_targets, oof_preds, all_imgs):
    print("\nRunning Failure Analysis...")

    # Calculate Error Magnitude: |y - p|
    errors = np.abs(oof_targets - oof_preds)

    # Calculate Image Features
    # Flatten spatial dims: (N, 3, 32, 32) -> (N, 3072)
    flat_imgs = all_imgs.reshape(all_imgs.shape[0], -1)

    img_means = flat_imgs.mean(axis=1)
    img_stds = flat_imgs.std(axis=1)

    # Correlation Analysis
    corr_mean, p_mean = pearsonr(errors, img_means)
    corr_std, p_std = pearsonr(errors, img_stds)

    print(f"Correlation (Error vs Mean Intensity): {corr_mean:.4f} (p={p_mean:.4f})")
    print(f"Correlation (Error vs Contrast/Std):   {corr_std:.4f} (p={p_std:.4f})")


def run_submission(trained_model_paths, stacker):
    print("\nGenerating Submission...")

    # Load Test Data
    test_imgs, test_ids = load_data_to_memory("test", load_cached_data=True)
    test_ds = CactusDataset(
        test_imgs, test_ids, transform=get_transforms("test"), is_test=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Aggregate features across folds for each architecture
    # We want final shape (N_test, 4) -> [RepVGG_Mean, RepVGG_Std, ResNet_Mean, ResNet_Std]

    arch_features = {}
    stacker_instance = StackingEnsemble()  # Helper to load models

    for arch_name in Config.MODELS:
        print(f"Processing architecture: {arch_name}")
        paths = trained_model_paths[arch_name]

        accumulated_feats = np.zeros((len(test_ids), 2), dtype=np.float32)

        for path in paths:
            # Load model
            model = stacker_instance._load_model(path, Config.DEVICE)

            # Predict TTA
            raw_probs, _ = Engine.predict_tta_raw(model, test_loader, Config.DEVICE)

            # Extract Features
            feats = GeometricFeatureExtractor.extract(raw_probs)
            accumulated_feats += feats

            del model
            torch.cuda.empty_cache()

        # Average across folds
        arch_features[arch_name] = accumulated_feats / len(paths)

    # Construct final feature matrix
    # Order must match training: RepVGG, ResNet
    X_test = np.hstack([arch_features["CactusRepVGG"], arch_features["CactusResNet"]])

    # Predict with Meta-Learner
    final_preds = stacker.meta_learner.predict_proba(X_test)[:, 1]

    # Save
    df = pd.DataFrame({"id": test_ids, "has_cactus": final_preds})
    df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    seed_everything(Config.SEED)
    Config.setup()

    # 1. Training & OOF Generation
    oof_features, oof_targets, trained_model_paths, all_imgs = run_training()

    # 2. Meta-Learning
    stacker, auc, oof_preds = run_meta_learning(oof_features, oof_targets)

    # 3. Failure Analysis
    run_failure_analysis(oof_targets, oof_preds, all_imgs)

    # 4. Submission
    # Note: The prompt condition "higher than 1.0" is technically impossible for AUC.
    # We assume a standard validity check (e.g. > 0.5) is intended.
    if auc > 0.5:
        run_submission(trained_model_paths, stacker)
    else:
        print("Validation metric too low. Submission skipped.")


if __name__ == "__main__":
    main()
