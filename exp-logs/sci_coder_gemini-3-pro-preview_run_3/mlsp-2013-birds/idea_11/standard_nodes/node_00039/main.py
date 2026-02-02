import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2

from library.config import Config
from library.utils import seed_everything, calculate_roc_auc, parse_label_string
from library.data import make_folds, get_loaders, get_test_loader
from library.engine import train_fold, validate, inference, save_submission
from library.models import get_model


def run():
    # 1. Setup
    seed_everything(Config.SEED)
    Config.setup()

    # 2. Data Preparation
    # make_folds loads train and val metadata, merges them, and creates folds
    # This ensures we use all available labeled data for Cross-Validation
    df_folds = make_folds(load_cached_data=True)

    num_samples = len(df_folds)
    num_classes = Config.NUM_CLASSES

    # Storage for OOF Ensemble Predictions
    # We accumulate predictions from all architectures here
    ensemble_oof_preds = np.zeros((num_samples, num_classes), dtype=np.float32)
    ensemble_oof_counts = np.zeros((num_samples, 1), dtype=np.float32)

    # Extract targets for evaluation
    oof_targets = np.zeros((num_samples, num_classes), dtype=np.float32)
    for idx, row in df_folds.iterrows():
        oof_targets[idx] = parse_label_string(row["labels"])

    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 3. Training & OOF Generation
    # Iterate over each architecture in the heterogeneous ensemble
    for arch in Config.ARCHITECTURES:
        print(f"\n=== Processing Architecture: {arch} ===")

        # 5-Fold Cross Validation
        for fold_idx in range(Config.N_FOLDS):
            print(f"  Fold {fold_idx}/{Config.N_FOLDS - 1}")

            # Get DataLoaders for this fold
            train_loader, valid_loader = get_loaders(fold_idx, df_folds)

            # Train model and save the best checkpoint
            best_auc, ckpt_path = train_fold(
                fold_idx, arch, train_loader, valid_loader, device=device
            )

            # Load the best model weights for OOF inference
            model = get_model(arch, num_classes=num_classes, pretrained=False)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.to(device)

            # Generate predictions on the validation set
            criterion = torch.nn.BCEWithLogitsLoss()
            _, _, preds = validate(model, valid_loader, criterion, device)

            # Map predictions back to original DataFrame indices
            # valid_loader contains samples where df_folds['fold'] == fold_idx
            valid_indices = df_folds[df_folds["fold"] == fold_idx].index.values

            # Accumulate predictions
            if len(preds) == len(valid_indices):
                ensemble_oof_preds[valid_indices] += preds
                ensemble_oof_counts[valid_indices] += 1
            else:
                print(
                    f"    WARNING: Prediction size {len(preds)} mismatch with indices {len(valid_indices)}"
                )

            # Cleanup to free GPU memory
            del model
            torch.cuda.empty_cache()

    # 4. Evaluation
    # Average predictions across all models in the ensemble
    mask = ensemble_oof_counts > 0
    avg_oof_preds = np.zeros_like(ensemble_oof_preds)
    avg_oof_preds[mask[:, 0]] = (
        ensemble_oof_preds[mask[:, 0]] / ensemble_oof_counts[mask[:, 0]]
    )

    # Compute Final Validation Metric (AUC)
    final_metric = calculate_roc_auc(oof_targets, avg_oof_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate Mean Absolute Error per sample
    per_sample_error = np.mean(np.abs(oof_targets - avg_oof_preds), axis=1)

    # Compute spectrogram statistics (Mean and Std) for correlation analysis
    img_means = []
    img_stds = []

    for idx, row in df_folds.iterrows():
        wav_filename = os.path.basename(row["file_path"])
        bmp_filename = wav_filename.replace(".wav", ".bmp")
        img_path = os.path.join(Config.SPECTROGRAM_DIR, bmp_filename)

        mu, sigma = 0.0, 0.0
        if os.path.exists(img_path):
            try:
                # Read image as grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    mu = np.mean(img)
                    sigma = np.std(img)
            except Exception:
                pass

        img_means.append(mu)
        img_stds.append(sigma)

    img_means = np.array(img_means)
    img_stds = np.array(img_stds)

    # Calculate correlations
    if np.std(img_means) > 1e-6:
        corr_mean = np.corrcoef(per_sample_error, img_means)[0, 1]
    else:
        corr_mean = 0.0

    if np.std(img_stds) > 1e-6:
        corr_std = np.corrcoef(per_sample_error, img_stds)[0, 1]
    else:
        corr_std = 0.0

    print(f"Correlation (Error vs Spectrogram Mean): {corr_mean:.6f}")
    print(f"Correlation (Error vs Spectrogram Std): {corr_std:.6f}")

    # 6. Submission
    THRESHOLD = 0.9072993371210134

    if final_metric > THRESHOLD:
        print("\nMetric threshold passed. Generating submission...")
        test_loader = get_test_loader()

        ensemble_test_preds = None
        model_count = 0
        test_rec_ids = None

        # Iterate over all trained models (3 architectures * 5 folds)
        for arch in Config.ARCHITECTURES:
            for fold_idx in range(Config.N_FOLDS):
                ckpt_path = os.path.join(
                    Config.CHECKPOINT_DIR, f"{arch}_fold_{fold_idx}_best.pth"
                )

                if not os.path.exists(ckpt_path):
                    print(f"Warning: Checkpoint {ckpt_path} not found.")
                    continue

                # Perform inference
                rec_ids, preds = inference(arch, ckpt_path, test_loader, device)

                # Initialize or accumulate predictions
                if ensemble_test_preds is None:
                    ensemble_test_preds = np.zeros_like(preds)
                    test_rec_ids = rec_ids

                ensemble_test_preds += preds
                model_count += 1

        # Average and Save
        if model_count > 0:
            avg_test_preds = ensemble_test_preds / model_count
            save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
            save_submission(test_rec_ids, avg_test_preds, save_path)
        else:
            print("Error: No models available for inference.")
    else:
        print(
            f"\nMetric {final_metric} did not pass threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    run()
