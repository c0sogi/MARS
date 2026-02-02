import os
import sys
import contextlib
import numpy as np
import torch
import scipy.stats
import pandas as pd

# Import provided library modules
import library.config as config
from library.utils import seed_everything, rmse_score, load_checkpoint
from library.model import ASPPShallowUNet
from library.dataset import get_dataloaders
from library.train import train_model
from library.inference import generate_predictions, apply_d4_tta


# Context manager to suppress stdout/stderr during verbose training
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def validate_ensemble(seeds):
    """
    Performs validation using the ensemble of trained models with TTA.
    Returns global RMSE and data for failure analysis.
    """
    # Load validation data
    # We use the provided get_dataloaders function
    _, val_loader, _ = get_dataloaders(load_cached_data=True)

    # Load all trained models
    models = []
    for seed in seeds:
        checkpoint_path = os.path.join(config.WORKING_DIR, f"model_seed_{seed}.pth")
        if os.path.exists(checkpoint_path):
            model = ASPPShallowUNet().to(config.DEVICE)
            # Load checkpoint
            load_checkpoint(checkpoint_path, model)
            model.eval()
            models.append(model)

    if not models:
        print("Error: No trained models found for validation.")
        return float("inf"), [], [], [], []

    all_preds = []
    all_targets = []
    all_noisy_means = []
    all_noisy_stds = []

    # Inference loop
    with torch.no_grad():
        for batch in val_loader:
            noisy, clean, _ = batch
            noisy = noisy.to(config.DEVICE)

            # Calculate input stats for failure analysis
            noisy_np = noisy.cpu().numpy()
            all_noisy_means.append(np.mean(noisy_np))
            all_noisy_stds.append(np.std(noisy_np))

            # Ensemble Prediction with TTA
            final_pred = None
            for model in models:
                # apply_d4_tta handles the 8-view augmentation and averaging for one model
                tta_pred = apply_d4_tta(model, noisy)

                if final_pred is None:
                    final_pred = tta_pred
                else:
                    final_pred += tta_pred

            # Average across ensemble members
            final_pred /= len(models)

            # Flatten and store
            all_preds.append(final_pred.cpu().numpy().flatten())
            all_targets.append(clean.numpy().flatten())

    # Calculate Global RMSE
    y_pred_flat = np.concatenate(all_preds)
    y_true_flat = np.concatenate(all_targets)
    score = rmse_score(y_true_flat, y_pred_flat)

    return score, all_preds, all_targets, all_noisy_means, all_noisy_stds


def analyze_failures(preds, targets, means, stds):
    """
    Analyzes the correlation between error and input features.
    """
    img_rmses = []

    for p, t in zip(preds, targets):
        mse = np.mean((p - t) ** 2)
        rmse = np.sqrt(mse)
        img_rmses.append(rmse)

    # Calculate correlations
    if len(img_rmses) > 1:
        corr_mean = scipy.stats.pearsonr(img_rmses, means)[0]
        corr_std = scipy.stats.pearsonr(img_rmses, stds)[0]

        print("-" * 30)
        print("Failure Analysis")
        print("-" * 30)
        print(f"Correlation (RMSE vs Input Mean Intensity): {corr_mean:.10f}")
        print(f"Correlation (RMSE vs Input Std Dev): {corr_std:.10f}")
    else:
        print("Insufficient data for failure analysis.")


def main():
    # Ensure reproducibility
    seed_everything(42)

    print("Starting execution...")

    # 1. Train Models
    # We train 5 models as per the Idea configuration.
    # Suppressing output to avoid cluttering logs with 5000 epochs of print statements.
    trained_seeds = []
    print(f"Training models for seeds: {config.SEEDS}")

    for seed in config.SEEDS:
        print(f"Training seed {seed}...")
        try:
            # Using suppress_output to hide the per-epoch logs from library.train
            with suppress_output():
                train_model(seed, load_cached_data=True)
            trained_seeds.append(seed)
        except Exception as e:
            print(f"Failed to train seed {seed}: {e}")

    if not trained_seeds:
        print("All training attempts failed. Exiting.")
        return

    # 2. Validation
    print("Performing ensemble validation...")
    val_rmse, val_preds, val_targets, val_means, val_stds = validate_ensemble(
        trained_seeds
    )

    # REQUIRED OUTPUT: Final Validation Metric
    print(f"Final Validation Metric: {val_rmse}")

    # 3. Failure Analysis
    analyze_failures(val_preds, val_targets, val_means, val_stds)

    # 4. Submission
    # Threshold defined in the task
    THRESHOLD = 0.011870221132053216

    if val_rmse < THRESHOLD:
        print(
            f"Validation metric {val_rmse} is strictly lower than {THRESHOLD}. Generating submission..."
        )
        # generate_predictions handles loading models, TTA, and saving CSV
        with suppress_output():
            generate_predictions(seeds=trained_seeds, load_cached_data=True)
        print(f"Submission saved to {config.SUBMISSION_PATH}")
    else:
        print(
            f"Validation metric {val_rmse} is NOT lower than {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
