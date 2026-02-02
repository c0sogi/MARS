import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, calculate_log_loss
from library.dataset import process_data, get_fold_loaders, get_test_loader
from library.model import SEAHN
from library.engine import Trainer, predict

# -----------------------------------------------------------------------------
# Configuration Overrides for Fast Baseline Execution
# -----------------------------------------------------------------------------
# We override specific Config values to ensure the code completes quickly
# while still providing a meaningful baseline.
Config.NUM_EPOCHS = 30
Config.PATIENCE = 8


def run():
    # 1. Setup and Initialization
    seed_everything(Config.SEED)
    Config.make_dirs()
    device = torch.device(Config.DEVICE)

    print(f"Initializing SEA-HN Pipeline on device: {device}")
    print(f"Configuration: {Config.NUM_EPOCHS} Epochs, {Config.NUM_FOLDS} Folds")

    # 2. Data Loading
    # Load full dataset arrays to handle Cross-Validation splitting and Analysis
    print("Loading and processing data...")
    X_img, X_stats, y, _, _, test_ids = process_data(load_cached_data=True)

    # Initialize arrays to store OOF predictions and Test ensemble predictions
    oof_preds = np.zeros(len(y))
    test_preds_sum = np.zeros(len(test_ids))

    # Define Stratified K-Fold splitter
    # Must use same parameters as in library.dataset to ensure index alignment
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # 3. Cross-Validation Training Loop
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_img, y)):
        print(f"\n" + "=" * 40)
        print(f"Training Fold {fold_idx + 1}/{Config.NUM_FOLDS}")
        print("=" * 40)

        # Get DataLoaders for this specific fold
        train_loader, val_loader = get_fold_loaders(fold_idx, load_cached_data=True)

        # Initialize Model
        model = SEAHN().to(device)

        # Initialize Optimizer (Adam)
        optimizer = optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Initialize Scheduler (ReduceLROnPlateau)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
            min_lr=Config.SCHEDULER_MIN_LR,
        )

        # Initialize Trainer
        trainer = Trainer(model, device, optimizer, scheduler)

        # Train the model
        best_loss = trainer.fit(
            train_loader, val_loader, epochs=Config.NUM_EPOCHS, patience=Config.PATIENCE
        )

        # --- Inference ---

        # 1. Validation Inference (OOF)
        # trainer.model contains the best weights from the run
        print("Generating validation predictions...")
        val_preds_fold = predict(trainer.model, val_loader, device)
        oof_preds[val_idx] = val_preds_fold

        # 2. Test Inference (Ensemble Component)
        # Load test loader (scales data based on full training set)
        print("Generating test predictions...")
        test_loader, _ = get_test_loader(load_cached_data=True)
        test_preds_fold = predict(trainer.model, test_loader, device)
        test_preds_sum += test_preds_fold

        # Cleanup to free GPU memory
        del model, optimizer, scheduler, trainer, train_loader, val_loader, test_loader
        torch.cuda.empty_cache()

    # 4. Validation Assessment
    print("\n" + "=" * 40)
    print("Validation Assessment")
    print("=" * 40)

    final_log_loss = calculate_log_loss(y, oof_preds)
    print(f"Final Validation Metric: {final_log_loss}")

    # 5. Failure Analysis
    print("\n" + "=" * 40)
    print("Failure Analysis")
    print("=" * 40)

    # Calculate error magnitude per sample
    errors = np.abs(y - oof_preds)

    # Define feature names corresponding to the stats vector construction in dataset.py
    # Order: [Mean, Std, Min, Max, Median] for Band 1, Band 2, Avg Band, then Inc Angle
    feature_names = [
        "b1_mean",
        "b1_std",
        "b1_min",
        "b1_max",
        "b1_median",
        "b2_mean",
        "b2_std",
        "b2_min",
        "b2_max",
        "b2_median",
        "avg_mean",
        "avg_std",
        "avg_min",
        "avg_max",
        "avg_median",
        "inc_angle",
    ]

    correlations = {}
    for i, name in enumerate(feature_names):
        # Extract feature column
        feat_values = X_stats[:, i]

        # Calculate Pearson correlation with error magnitude
        # Handle constant features (std=0) to avoid division by zero
        if np.std(feat_values) > 1e-9:
            corr, _ = pearsonr(feat_values, errors)
            correlations[name] = corr
        else:
            correlations[name] = 0.0

    # Sort features by absolute correlation strength
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Correlation between Input Features and Error Magnitude (Top 5):")
    for name, corr in sorted_corrs[:5]:
        print(f"  {name:<12}: {corr:.6f}")

    # 6. Submission Generation
    threshold = 0.17493283735739185

    if final_log_loss < threshold:
        print(f"\nValidation metric ({final_log_loss}) meets threshold ({threshold}).")
        print("Generating submission file...")

        # Average predictions across folds
        avg_test_preds = test_preds_sum / Config.NUM_FOLDS

        # Create submission DataFrame
        submission = pd.DataFrame({"id": test_ids, "is_iceberg": avg_test_preds})

        # Save
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation metric ({final_log_loss}) does NOT meet threshold ({threshold})."
        )
        print("Submission generation skipped.")


if __name__ == "__main__":
    run()
