import os
import sys
import pandas as pd
import numpy as np
import torch
import soundfile as sf
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler
from scipy.stats import pearsonr
import time

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, calculate_metrics
from library.architectures import WhaleClassifier
from library.data_factory import WhaleDataset
from library.trainer import ModelTrainer
from library.inference import Predictor

# =============================================================================
# Configuration Overrides for Fast Baseline
# =============================================================================
# Limit epochs to ensure execution within time limits while maintaining performance.
# Pre-trained models converge quickly on spectrograms.
Config.EPOCHS = 4
Config.BATCH_SIZE = 128
Config.NUM_WORKERS = 4


def train_ensemble():
    """
    Orchestrates the training of the 5-fold 4-config ensemble.
    Trains ONLY on the training set (train.csv) to preserve val.csv as hold-out.
    """
    print("Loading training metadata...")
    # Cite debug_lesson_12: Strictly Segregate Hold-Out Data
    train_df = pd.read_csv(Config.TRAIN_CSV)

    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Stratified K-Fold on TRAIN data only
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Loop through Folds
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df["label"])):
        print(f"\n{'='*40}\nStarting Fold {fold}/{Config.N_FOLDS - 1}\n{'='*40}")

        # Split Dataframes
        fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)

        # Create Datasets
        train_dataset = WhaleDataset(fold_train_df, phase="train")
        val_dataset = WhaleDataset(fold_val_df, phase="val")

        # Create Sampler for Class Imbalance
        targets = fold_train_df["label"].values
        class_counts = np.bincount(targets.astype(int))
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = class_weights[targets.astype(int)]
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(sample_weights),
            replacement=True,
        )

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            sampler=sampler,
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

        # Train each Configuration
        for config in Config.ENSEMBLE_CONFIGS:
            save_name = f"{config['name']}_fold_{fold}"
            checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"{save_name}.pth")

            if os.path.exists(checkpoint_path):
                print(f"Checkpoint {save_name} already exists. Skipping training.")
                continue

            print(
                f"Training Config: {config['name']} (Objective: {config['objective']})"
            )

            # Initialize Model
            model = WhaleClassifier(config["arch"], pretrained=True)
            model.to(device)

            # Optimizer & Scheduler
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
            )

            # Trainer
            trainer = ModelTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                objective=config["objective"],
                save_name=save_name,
            )

            # Fit
            trainer.fit(epochs=Config.EPOCHS, patience=Config.PATIENCE)

            # Cleanup to save memory
            del model, optimizer, scheduler, trainer
            torch.cuda.empty_cache()


def perform_failure_analysis(val_df, y_true, y_pred):
    """
    Analyzes model failures by correlating error magnitude with input signal statistics.
    """
    print("\n--- Failure Analysis ---")

    # Calculate Error Magnitude
    errors = np.abs(y_true - y_pred)

    # Extract Features (RMS, Peak) for Validation Set
    print("Extracting signal statistics for validation set...")
    rms_values = []
    peak_values = []

    for idx, row in val_df.iterrows():
        file_path = os.path.join(Config.INPUT_ROOT, row["file_path"])
        try:
            data, _ = sf.read(file_path)
            # Handle multi-channel
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            rms = np.sqrt(np.mean(data**2))
            peak = np.max(np.abs(data))

            rms_values.append(rms)
            peak_values.append(peak)
        except Exception:
            rms_values.append(0)
            peak_values.append(0)

    rms_values = np.array(rms_values)
    peak_values = np.array(peak_values)

    # Calculate Correlations
    corr_rms, _ = pearsonr(errors, rms_values)
    corr_peak, _ = pearsonr(errors, peak_values)

    print(f"Correlation between Error and Signal RMS: {corr_rms:.4f}")
    print(f"Correlation between Error and Signal Peak: {corr_peak:.4f}")

    # High error analysis
    high_error_indices = np.argsort(errors)[-5:]
    print("\nTop 5 High Error Samples:")
    for idx in high_error_indices:
        print(
            f"File: {val_df.iloc[idx]['file_path']}, True: {y_true[idx]}, Pred: {y_pred[idx]:.4f}, Error: {errors[idx]:.4f}"
        )


def main():
    seed_everything(Config.SEED)

    # 1. Train the Ensemble
    train_ensemble()

    # 2. Inference & Meta-Learning
    print("\nStarting Inference Pipeline...")
    predictor = Predictor(debug=False)

    # Clear stale cache if it exists to ensure we use new models/data split
    # Cite debug_lesson_4
    for cache_file in ["oof_features.npy", "val_features.npy", "test_features.npy"]:
        cache_path = os.path.join(Config.CACHE_DIR, cache_file)
        if os.path.exists(cache_path):
            print(f"Removing stale cache: {cache_path}")
            os.remove(cache_path)

    # Generate OOF predictions for TRAIN set
    print("Generating OOF predictions for Training Set...")
    X_train_oof, y_train_oof = predictor.generate_oof_predictions(load_cached_data=True)

    # Train Meta-Learner on Train OOFs
    from sklearn.linear_model import LogisticRegression

    print("Training Meta-Learner on Train OOFs...")
    meta_learner = LogisticRegression(random_state=Config.SEED, solver="liblinear")
    meta_learner.fit(X_train_oof, y_train_oof)

    # 3. Validation Assessment (Hold-out)
    print("\nEvaluating on Hold-out Validation Set...")
    val_df = pd.read_csv(Config.VAL_CSV)

    # Generate features for Validation set using Bagging (same as Test set)
    # This ensures no leakage: models trained on Train predict on Val
    X_val_meta = predictor.generate_bagged_features(
        val_df, cache_name="val_features.npy"
    )
    y_val_true = val_df["label"].values

    # Predict using Meta-Learner
    # Handle NaNs if any (though bagging usually handles this)
    X_val_meta = np.nan_to_num(X_val_meta, nan=0.5)
    y_val_pred = meta_learner.predict_proba(X_val_meta)[:, 1]

    # Calculate Metric
    final_metric = calculate_metrics(y_val_true, y_val_pred)
    print(f"Final Validation Metric: {final_metric:.16f}")

    # 4. Failure Analysis
    perform_failure_analysis(val_df, y_val_true, y_val_pred)

    # 5. Submission
    THRESHOLD = 0.90  # Lower threshold for safety, aim for best possible

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric:.6f}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        # Pass the trained meta-learner to create_submission to avoid retraining logic issues
        predictor.create_submission(meta_learner_override=meta_learner)
    else:
        print(
            f"\nMetric ({final_metric:.6f}) did not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
