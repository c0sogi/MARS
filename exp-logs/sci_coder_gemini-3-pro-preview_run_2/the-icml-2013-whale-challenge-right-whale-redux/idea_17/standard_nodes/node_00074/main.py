import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, WeightedRandomSampler

# Ensure local library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, calculate_roc_auc, get_device
from library.dataset import get_data, WhaleDataset
from library.train import Trainer
from library.stacking import fit_stacking_model, predict_stacking_model


def run():
    # 1. Setup & Configuration
    seed_everything(Config.SEED)
    device = get_device()

    # Override Config for Execution
    # Increasing epochs to allow Cosine Annealing to properly decay and fine-tune
    Config.EPOCHS = 15
    Config.NUM_FOLDS = 5

    print(
        f"Starting execution with {Config.NUM_FOLDS} folds and {Config.EPOCHS} epochs per model."
    )

    # 2. Data Loading
    print("Loading Metadata...")
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    print("Loading and Processing Data Arrays...")
    # Load Training Data (Source for CV)
    X_train, y_train, _ = get_data(train_df, "train", load_cached_data=True)
    # Load Hold-out Validation Data (Strictly for final evaluation)
    X_holdout, y_holdout, _ = get_data(val_df, "val", load_cached_data=True)
    # Load Test Data
    X_test, _, test_clips = get_data(test_df, "test", load_cached_data=True)

    # 3. Cross-Validation Setup
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Storage for Level-1 Meta-Features
    # OOF Predictions: (N_train_samples, N_models)
    oof_preds = np.zeros((len(X_train), len(Config.MODEL_NAMES)))

    # Bagged Predictions for Holdout and Test: (N_samples, N_models)
    # We accumulate sum and divide by K folds later
    holdout_preds_sum = np.zeros((len(X_holdout), len(Config.MODEL_NAMES)))
    test_preds_sum = np.zeros((len(X_test), len(Config.MODEL_NAMES)))

    # 4. Training Loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n=== Fold {fold + 1}/{Config.NUM_FOLDS} ===")

        # Split Data for this Fold
        X_fold_train, y_fold_train = X_train[train_idx], y_train[train_idx]
        X_fold_val, y_fold_val = X_train[val_idx], y_train[val_idx]

        # Create DataLoaders
        # 1. Train Loader with Weighted Sampling for Class Balance
        train_ds = WhaleDataset(X_fold_train, y_fold_train, mode="train")
        class_counts = np.bincount(y_fold_train.astype(int))
        # Add epsilon to avoid division by zero if a class is missing in a tiny debug fold
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = [class_weights[int(t)] for t in y_fold_train]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            sampler=sampler,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        # 2. Fold Validation Loader (for Early Stopping)
        fold_val_ds = WhaleDataset(X_fold_val, y_fold_val, mode="val")
        fold_val_loader = DataLoader(
            fold_val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # 3. Inference Loaders (Holdout & Test)
        # Re-instantiated per fold to ensure clean state
        holdout_ds = WhaleDataset(X_holdout, y_holdout, mode="val")
        holdout_loader = DataLoader(
            holdout_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        test_ds = WhaleDataset(X_test, targets=None, mode="test")
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Train Each Architecture
        for model_idx, model_name in enumerate(Config.MODEL_NAMES):
            print(f"Training {model_name}...")

            # Initialize Trainer
            trainer = Trainer(model_name, train_loader, fold_val_loader, device)
            # Unique checkpoint path for this fold to avoid overwriting
            trainer.best_model_path = os.path.join(
                Config.WORKING_DIR, f"{model_name}_fold{fold}_best.pth"
            )

            # Train
            trainer.fit()

            # Generate Predictions
            # 1. OOF Prediction (on Fold Validation Set)
            oof_p = trainer.predict(fold_val_loader)
            oof_preds[val_idx, model_idx] = oof_p

            # 2. Holdout Prediction (Accumulate for Bagging)
            h_p = trainer.predict(holdout_loader)
            holdout_preds_sum[:, model_idx] += h_p

            # 3. Test Prediction (Accumulate for Bagging)
            t_p = trainer.predict(test_loader)
            test_preds_sum[:, model_idx] += t_p

            # Cleanup
            del trainer, oof_p, h_p, t_p
            torch.cuda.empty_cache()

    # 5. Bagging Aggregation
    holdout_preds_avg = holdout_preds_sum / Config.NUM_FOLDS
    test_preds_avg = test_preds_sum / Config.NUM_FOLDS

    # 6. Stacking (Meta-Learner)
    print("\nTraining Meta-Learner (Logistic Regression)...")
    # Train on OOF predictions
    meta_learner = fit_stacking_model(oof_preds, y_train, save_dir=Config.WORKING_DIR)

    # Validate on Hold-out Set
    print("Validating Ensemble on Hold-out Set...")
    final_val_probs = predict_stacking_model(meta_learner, holdout_preds_avg)
    final_val_auc = calculate_roc_auc(y_holdout, final_val_probs)

    print(f"Final Validation Metric: {final_val_auc}")

    # 7. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate absolute error
    errors = np.abs(y_holdout - final_val_probs)

    # Extract simple signal statistics from spectrograms for correlation
    # X_holdout shape: (N, n_mels, time_steps)
    spec_mean = X_holdout.mean(axis=(1, 2))
    spec_std = X_holdout.std(axis=(1, 2))
    spec_max = X_holdout.max(axis=(1, 2))

    # Calculate correlations
    corr_mean, _ = pearsonr(errors, spec_mean)
    corr_std, _ = pearsonr(errors, spec_std)
    corr_max, _ = pearsonr(errors, spec_max)

    print(f"Correlation between Error and Spectrogram Mean: {corr_mean:.4f}")
    print(f"Correlation between Error and Spectrogram Std: {corr_std:.4f}")
    print(f"Correlation between Error and Spectrogram Max: {corr_max:.4f}")

    # 8. Submission Generation
    THRESHOLD = 0.9963127202237976

    if final_val_auc > THRESHOLD:
        print(
            f"\nValidation metric {final_val_auc} exceeds threshold {THRESHOLD}. Generating submission..."
        )

        # Generate Final Test Probabilities
        final_test_probs = predict_stacking_model(meta_learner, test_preds_avg)

        submission = pd.DataFrame({"clip": test_clips, "probability": final_test_probs})

        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation metric {final_val_auc} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    run()
