import os
import sys
import numpy as np
import pandas as pd
import torch
import glob
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

# Import from provided library files
from library.config import Config, set_seed, load_data_splits
from library.trainer import run_fold
from library.inference import run_inference, predict_with_tta
from library.models import get_model
from library.data import get_dataloaders
from library.utils import calculate_multilabel_auc


def main():
    # 1. Setup
    # Using 30 epochs as a fast baseline that fits within time limits
    config = Config(epochs=30, batch_size=32)
    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Execution Device: {device}")

    # 2. Data Loading
    print("Loading Data Splits...")
    folds_df, test_df = load_data_splits(config, load_cached_data=True)

    # 3. Training Loop
    # Train all architectures on all folds
    print("\n=== Starting Training Phase ===")
    for model_name in config.ARCHITECTURES:
        for fold_idx in range(config.N_FOLDS):
            print(f"\nTraining {model_name} - Fold {fold_idx}")
            run_fold(fold_idx, model_name, config, folds_df, test_df)

    # 4. Validation & OOF Inference
    print("\n=== Starting Validation & OOF Inference ===")

    oof_preds = []
    oof_targets = []
    oof_rec_ids = []

    # Features for failure analysis
    fa_image_means = []
    fa_image_stds = []
    fa_num_labels = []

    for fold_idx in range(config.N_FOLDS):
        print(f"Generating OOF predictions for Fold {fold_idx}...")

        # Get validation loader for this fold
        _, val_loader, _ = get_dataloaders(fold_idx, folds_df, test_df, config)

        # Store targets and features for this fold
        fold_targets = []
        fold_image_means = []
        fold_image_stds = []

        # We need to iterate the loader once to get targets and compute image stats
        # Note: We can't just take val_loader.dataset.labels because we need to ensure alignment with batches
        # However, since shuffle=False for val_loader, order is preserved.

        # Let's do a pass to collect ground truth and image stats
        for images, labels in val_loader:
            fold_targets.append(labels.numpy())

            # Compute image stats for failure analysis
            # images: (B, 3, 224, 224)
            # Mean across H, W, C
            imgs_np = images.numpy()
            fold_image_means.extend(np.mean(imgs_np, axis=(1, 2, 3)))
            fold_image_stds.extend(np.std(imgs_np, axis=(1, 2, 3)))

        fold_targets = np.concatenate(fold_targets, axis=0)
        oof_targets.append(fold_targets)

        fa_image_means.extend(fold_image_means)
        fa_image_stds.extend(fold_image_stds)

        # Calculate number of labels per sample
        fa_num_labels.extend(np.sum(fold_targets, axis=1))

        # Ensemble Prediction for this fold
        # Average predictions from all models and their top checkpoints for this fold
        fold_ensemble_probs = np.zeros_like(fold_targets, dtype=np.float64)
        model_count = 0

        for model_name in config.ARCHITECTURES:
            checkpoint_base_dir = os.path.join(
                config.OUTPUT_DIR, "checkpoints", model_name
            )
            search_pattern = os.path.join(
                checkpoint_base_dir, f"{model_name}_fold_{fold_idx}_*.pth"
            )
            checkpoints = glob.glob(search_pattern)

            for ckpt_path in checkpoints:
                model = get_model(model_name, config, device=device)
                try:
                    checkpoint = torch.load(ckpt_path, map_location=device)
                    model.load_state_dict(checkpoint["model_state_dict"])
                except Exception as e:
                    print(f"Error loading {ckpt_path}: {e}")
                    continue

                # Predict
                model_probs = []
                with torch.no_grad():
                    for images, _ in val_loader:
                        images = images.to(device)
                        batch_probs = predict_with_tta(model, images, device)
                        model_probs.append(batch_probs.cpu().numpy())

                fold_ensemble_probs += np.concatenate(model_probs, axis=0)
                model_count += 1

                del model
                torch.cuda.empty_cache()

        if model_count > 0:
            fold_ensemble_probs /= model_count

        oof_preds.append(fold_ensemble_probs)

    # Concatenate all OOF results
    y_true = np.concatenate(oof_targets, axis=0)
    y_pred = np.concatenate(oof_preds, axis=0)

    # 5. Metric Calculation
    final_metric = calculate_multilabel_auc(y_true, y_pred)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")

    # Calculate error magnitude per sample (L1 error averaged across classes)
    # shape: (N_samples,)
    error_per_sample = np.mean(np.abs(y_true - y_pred), axis=1)

    # Convert lists to arrays
    fa_image_means = np.array(fa_image_means)
    fa_image_stds = np.array(fa_image_stds)
    fa_num_labels = np.array(fa_num_labels)

    # Correlations
    corr_mean, _ = pearsonr(error_per_sample, fa_image_means)
    corr_std, _ = pearsonr(error_per_sample, fa_image_stds)
    corr_labels, _ = pearsonr(error_per_sample, fa_num_labels)

    print(f"Correlation (Error vs Image Mean Intensity): {corr_mean:.4f}")
    print(f"Correlation (Error vs Image Contrast/Std): {corr_std:.4f}")
    print(f"Correlation (Error vs Number of Labels): {corr_labels:.4f}")

    # 7. Submission
    threshold = 0.9479806884980326
    if final_metric > threshold:
        print(
            f"\nMetric ({final_metric}) > Threshold ({threshold}). Generating submission..."
        )
        run_inference()
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
