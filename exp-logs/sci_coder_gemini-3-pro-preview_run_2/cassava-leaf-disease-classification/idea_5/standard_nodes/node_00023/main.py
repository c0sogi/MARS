import os
import sys
import numpy as np
import pandas as pd
import torch
import scipy.stats as stats
from sklearn.metrics import accuracy_score

# Import from library
from library.config import Config
from library.utils import seed_everything
from library.data import prepare_folds, get_dataloaders
from library.model import get_model
from library.engine import train_fold
from library.inference import ensemble_inference


def main():
    # ------------------------------------------------------------------
    # 1. Configuration & Setup
    # ------------------------------------------------------------------
    # Initialize config instance
    cfg = Config()

    # Set seeds for reproducibility
    seed_everything(cfg.seed)

    print(f"Running with Device: {cfg.device}")
    print(f"Folds: {cfg.n_folds}")
    print(f"Epochs per Fold: {cfg.epochs}")

    # ------------------------------------------------------------------
    # 2. Data Preparation
    # ------------------------------------------------------------------
    print("Preparing stratified folds...")
    # Force regeneration of folds to ensure consistency with current run
    df_folds = prepare_folds(load_cached_data=False)

    # ------------------------------------------------------------------
    # 3. Training Loop (5-Fold CV)
    # ------------------------------------------------------------------
    print("\nStarting 5-Fold Cross-Validation Training...")
    for fold in range(cfg.n_folds):
        train_fold(fold, cfg)

    # ------------------------------------------------------------------
    # 4. OOF Validation & Metric Calculation
    # ------------------------------------------------------------------
    print("\nStarting Out-Of-Fold (OOF) Validation...")

    oof_preds_list = []
    oof_targets_list = []
    oof_file_paths = []

    device = cfg.device

    for fold in range(cfg.n_folds):
        print(f"Validating Fold {fold}...")

        # Get dataloader for this fold
        _, val_loader = get_dataloaders(fold, cfg)

        # Load the best model for this fold
        model_path = os.path.join(cfg.working_dir, f"fold_{fold}_best.pth")
        if not os.path.exists(model_path):
            print(f"Error: Model for fold {fold} not found at {model_path}!")
            continue

        model = get_model(cfg, pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        fold_probs = []
        fold_targets = []

        # Collect file paths for failure analysis
        # val_loader.dataset is a CassavaDataset, which has .df
        val_df = val_loader.dataset.df
        oof_file_paths.extend(val_df["file_path"].tolist())

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)

                # Forward pass
                logits = model(images)
                # Convert to probabilities
                probs = torch.softmax(logits, dim=1)

                fold_probs.append(probs.cpu().numpy())
                fold_targets.append(targets.numpy())

        oof_preds_list.append(np.concatenate(fold_probs, axis=0))
        oof_targets_list.append(np.concatenate(fold_targets, axis=0))

        # Cleanup to save memory
        del model
        torch.cuda.empty_cache()

    # Concatenate all folds
    if len(oof_preds_list) > 0:
        oof_preds = np.concatenate(oof_preds_list, axis=0)
        oof_targets = np.concatenate(oof_targets_list, axis=0)

        # Compute Final Metric (Accuracy)
        pred_labels = np.argmax(oof_preds, axis=1)
        final_metric = accuracy_score(oof_targets, pred_labels)

        print(f"Final Validation Metric: {final_metric}")
    else:
        print("Error: No predictions generated.")
        final_metric = 0.0

    # ------------------------------------------------------------------
    # 5. Failure Analysis
    # ------------------------------------------------------------------
    print("\nPerforming Failure Analysis...")

    if len(oof_preds_list) > 0:
        # Calculate Error Magnitude (1 - probability of the true class)
        row_indices = np.arange(len(oof_targets))
        true_class_probs = oof_preds[row_indices, oof_targets]
        error_magnitudes = 1.0 - true_class_probs

        # Extract File Sizes
        file_sizes = []
        valid_indices = []

        for i, rel_path in enumerate(oof_file_paths):
            full_path = os.path.join(cfg.input_dir, rel_path)
            if os.path.exists(full_path):
                try:
                    size = os.path.getsize(full_path)
                    file_sizes.append(size)
                    valid_indices.append(i)
                except:
                    pass

        if len(file_sizes) > 100:
            # Filter errors to match valid file sizes
            valid_errors = error_magnitudes[valid_indices]
            valid_sizes = np.array(file_sizes)

            # Calculate Pearson Correlation
            correlation = np.corrcoef(valid_errors, valid_sizes)[0, 1]
            print(f"Correlation between Error Magnitude and File Size: {correlation}")
        else:
            print("Insufficient data for correlation analysis.")

    # ------------------------------------------------------------------
    # 6. Submission Generation
    # ------------------------------------------------------------------
    threshold = 0.8995994659546062

    if final_metric > threshold:
        print(
            f"\nMetric ({final_metric}) > Threshold ({threshold}). Generating Submission..."
        )
        ensemble_inference(cfg)
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({threshold}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
