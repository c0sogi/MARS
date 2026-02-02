import sys
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr

# Import library components
from library.config import Config
from library.utils import set_seed, save_submission, load_checkpoint
from library.data_loader import process_and_cache_data, get_cv_loaders, get_test_loader
from library.train_eval import fit_fold
from library.model import SimpleCNN


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration and Setup
    # --------------------------------------------------------------------------
    Config.MAX_SAMPLES = None  # Use full dataset to ensure metric quality

    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print("Configuration:")
    print(f"  Device: {device}")
    print(f"  Epochs: {Config.NUM_EPOCHS}")
    print(f"  Folds: {Config.NUM_FOLDS}")

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("\nLoading and Processing Data...")
    # Load full training data (train + val combined for CV) and test data
    X, y, angles, X_test, ids_test, angles_test = process_and_cache_data(
        load_cached_data=True
    )

    print(f"  Training Data Shape: {X.shape}")
    print(f"  Test Data Shape: {X_test.shape}")

    # --------------------------------------------------------------------------
    # 3. Cross-Validation Loop
    # --------------------------------------------------------------------------
    # Storage for Out-Of-Fold predictions and Test predictions
    oof_preds = np.zeros(len(y))
    test_preds_accum = np.zeros((len(ids_test), Config.NUM_FOLDS))

    # Re-create StratifiedKFold to map predictions back to original indices
    # This ensures we fill oof_preds in the correct locations
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        print(f"\n--- Processing Fold {fold} ---")

        # Get DataLoaders
        train_loader, val_loader = get_cv_loaders(fold, X, y, angles)

        # Train the model
        fit_fold(fold, train_loader, val_loader)

        # ----------------------------------------------------------------------
        # Inference
        # ----------------------------------------------------------------------
        print(f"  Running Inference for Fold {fold}...")

        # Load Best Model
        model = SimpleCNN().to(device)
        ckpt_path = os.path.join(Config.WORKING_DIR, f"model_best_fold_{fold}.pth")
        load_checkpoint(ckpt_path, model)
        model.eval()

        # 1. Validation Inference (OOF)
        val_probs = []
        with torch.no_grad():
            for images, angs, _ in val_loader:
                images = images.to(device)
                angs = angs.to(device)
                out = model(images, angs)
                probs = torch.sigmoid(out).cpu().numpy().flatten()
                val_probs.extend(probs)

        # Store OOF predictions
        # val_loader iterates sequentially over the subset defined by val_idx
        oof_preds[val_idx] = val_probs

        # 2. Test Inference
        test_loader = get_test_loader(X_test, angles_test, ids_test)
        fold_test_probs = []
        with torch.no_grad():
            for images, angs in test_loader:
                images = images.to(device)
                angs = angs.to(device)
                out = model(images, angs)
                probs = torch.sigmoid(out).cpu().numpy().flatten()
                fold_test_probs.extend(probs)

        test_preds_accum[:, fold] = fold_test_probs

    # --------------------------------------------------------------------------
    # 4. Evaluation and Failure Analysis
    # --------------------------------------------------------------------------
    print("\n--- Evaluation ---")

    # Clip predictions for numerical stability in Log Loss
    oof_preds_clipped = np.clip(oof_preds, 1e-15, 1 - 1e-15)
    final_metric = log_loss(y, oof_preds_clipped)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate absolute error
    errors = np.abs(y - oof_preds)

    # Extract simple features for correlation
    # X shape: (N, 3, 75, 75). Channel 0 is Band 1 (HH), Channel 1 is Band 2 (HV).
    b1_means = X[:, 0, :, :].mean(axis=(1, 2))
    b2_means = X[:, 1, :, :].mean(axis=(1, 2))
    b1_stds = X[:, 0, :, :].std(axis=(1, 2))

    # Compute correlations
    correlations = {
        "Incidence Angle": pearsonr(errors, angles)[0],
        "Band 1 Mean": pearsonr(errors, b1_means)[0],
        "Band 2 Mean": pearsonr(errors, b2_means)[0],
        "Band 1 Std": pearsonr(errors, b1_stds)[0],
    }

    print("Correlation between Error Magnitude and Features:")
    for feature, corr in correlations.items():
        print(f"  {feature}: {corr:.4f}")

    # --------------------------------------------------------------------------
    # 5. Submission
    # --------------------------------------------------------------------------
    threshold = 0.18145903282502943

    if final_metric < threshold:
        print(
            f"\nMetric ({final_metric:.6f}) passed threshold ({threshold:.6f}). Generating submission..."
        )

        # Average predictions across folds
        avg_test_preds = test_preds_accum.mean(axis=1)

        # Save submission
        save_submission(avg_test_preds, ids_test)
    else:
        print(
            f"\nMetric ({final_metric:.6f}) did not pass threshold ({threshold:.6f}). Submission skipped."
        )


if __name__ == "__main__":
    main()
