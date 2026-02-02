import sys
import os
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

# Ensure library modules can be imported
sys.path.append("./library")

from library.config import Config
from library.utils import set_seed, calculate_metric
from library.dataset import create_dataloaders, create_test_dataloader
from library.model import ResNet18DualPool
from library.trainer import Trainer


def main():
    # 1. Configuration & Setup
    config = Config(debug=False)
    set_seed(config.seed)

    print(f"Initializing K-Fold Training (K={config.n_folds})...")

    # Storage for Global Validation & Analysis
    all_oof_preds = []
    all_oof_targets = []
    all_oof_pixel_means = []
    all_oof_pixel_stds = []

    # 2. K-Fold Training Loop
    for fold in range(config.n_folds):
        print(f"\n--- Fold {fold} ---")

        # Create DataLoaders
        train_loader, val_loader = create_dataloaders(config, fold_idx=fold)

        # Initialize Model
        model = ResNet18DualPool(config)

        # Initialize Trainer
        trainer = Trainer(config, model, train_loader, val_loader, fold_idx=fold)

        # Train
        trainer.fit()

        # --- OOF Inference & Metadata Collection ---
        print(f"Generating OOF predictions for Fold {fold}...")

        # Load best model weights for this fold
        model.load_state_dict(torch.load(trainer._get_model_path()))
        model.eval()
        model.to(config.device)

        fold_preds = []
        fold_targets = []
        fold_pixel_means = []
        fold_pixel_stds = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(config.device)

                # Predict
                outputs = model(images)
                probs = torch.sigmoid(outputs)

                fold_preds.append(probs.cpu().numpy())
                fold_targets.append(labels.cpu().numpy())

                # Compute Image Stats for Failure Analysis (on CPU)
                # images is (B, 3, H, W). Normalized.
                imgs_np = images.cpu().numpy()
                # Mean/Std over C, H, W per sample
                means = np.mean(imgs_np, axis=(1, 2, 3))
                stds = np.std(imgs_np, axis=(1, 2, 3))

                fold_pixel_means.append(means)
                fold_pixel_stds.append(stds)

        all_oof_preds.append(np.concatenate(fold_preds))
        all_oof_targets.append(np.concatenate(fold_targets))
        all_oof_pixel_means.append(np.concatenate(fold_pixel_means))
        all_oof_pixel_stds.append(np.concatenate(fold_pixel_stds))

    # 3. Global Validation Metric
    y_pred_oof = np.concatenate(all_oof_preds)
    y_true_oof = np.concatenate(all_oof_targets)

    final_metric = calculate_metric(y_true_oof, y_pred_oof)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate Mean Squared Error per sample (averaged over classes)
    sample_errors = np.mean((y_pred_oof - y_true_oof) ** 2, axis=1)

    pixel_means = np.concatenate(all_oof_pixel_means)
    pixel_stds = np.concatenate(all_oof_pixel_stds)

    # Correlations
    if np.std(sample_errors) > 1e-9 and np.std(pixel_means) > 1e-9:
        corr_mean, _ = pearsonr(sample_errors, pixel_means)
    else:
        corr_mean = 0.0

    if np.std(sample_errors) > 1e-9 and np.std(pixel_stds) > 1e-9:
        corr_std, _ = pearsonr(sample_errors, pixel_stds)
    else:
        corr_std = 0.0

    print(f"Correlation (Error vs Input Mean): {corr_mean:.4f}")
    print(f"Correlation (Error vs Input Std): {corr_std:.4f}")

    # 5. Submission
    threshold = 0.8739452549958209
    if final_metric > threshold:
        print("\nMetric threshold passed. Generating submission...")

        # Load Test Data
        test_loader = create_test_dataloader(config)

        # Accumulate predictions from all folds (Bagging Ensemble)
        avg_preds = None

        for fold in range(config.n_folds):
            # Load Model
            model = ResNet18DualPool(config)
            model_path = os.path.join(config.model_output_dir, f"model_fold_{fold}.pth")
            model.load_state_dict(torch.load(model_path))
            model.to(config.device)
            model.eval()

            fold_test_preds = []

            with torch.no_grad():
                for images, _ in test_loader:
                    images = images.to(config.device)
                    outputs = model(images)
                    probs = torch.sigmoid(outputs)
                    fold_test_preds.append(probs.cpu().numpy())

            fold_test_preds = np.concatenate(fold_test_preds)

            if avg_preds is None:
                avg_preds = fold_test_preds
            else:
                avg_preds += fold_test_preds

        # Average probabilities
        avg_preds /= config.n_folds

        # Format Submission
        # Retrieve rec_ids from the test dataset
        test_rec_ids = test_loader.dataset.df["rec_id"].values

        submission_rows = []
        for i, rec_id in enumerate(test_rec_ids):
            probs = avg_preds[i]  # Shape (19,)
            for species_idx, prob in enumerate(probs):
                # Id format: rec_id * 100 + species_idx
                row_id = int(rec_id * 100 + species_idx)
                submission_rows.append([row_id, prob])

        df_sub = pd.DataFrame(submission_rows, columns=["Id", "Probability"])

        # Sort by Id for consistency
        df_sub = df_sub.sort_values("Id")

        # Save
        df_sub.to_csv(config.submission_path, index=False)
        print(f"Submission saved to {config.submission_path}")

    else:
        print(
            f"\nMetric {final_metric} did not pass threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
