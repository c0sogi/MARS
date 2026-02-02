import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import from library
from library.config import Config
from library.utils import set_seed, calculate_roc_auc, print_metric
from library.data_loader import get_train_val_loaders, get_test_loader, load_data
from library.model_factory import WhaleEfficientNet
from library.trainer import run_fold


def main():
    # 1. Setup
    print("Initializing Run...")
    set_seed(Config.SEED)

    # Adjust Config for Fast Baseline execution
    Config.DEBUG = False  # Ensure we use full dataset for best score

    # Ensure directories exist
    Config.setup_directories()

    # Load data once to ensure cache is ready
    print("Ensuring data is loaded/processed...")
    load_data(load_cached_data=True)

    # 2. Cross-Validation Loop
    oof_preds = []
    oof_targets = []

    # We need to store validation data features for failure analysis
    val_features_list = []

    models = []

    for fold_idx in range(Config.N_FOLDS):
        print(f"\n----------------- Fold {fold_idx} -----------------")

        # Get loaders
        train_loader, val_loader = get_train_val_loaders(
            fold_idx, load_cached_data=True
        )

        # Train
        model, val_auc = run_fold(fold_idx, train_loader, val_loader)
        models.append(model)

        # Generate OOF predictions for this fold with the final model
        model.to(Config.DEVICE)
        model.eval()

        fold_preds = []
        fold_targets = []
        fold_features = []  # (mean, std, max)

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(Config.DEVICE)

                # Compute features for failure analysis
                # inputs: (B, 1, F, T)
                flat_inputs = inputs.view(inputs.size(0), -1)
                b_mean = flat_inputs.mean(dim=1).cpu().numpy()
                b_std = flat_inputs.std(dim=1).cpu().numpy()
                b_max = flat_inputs.max(dim=1).values.cpu().numpy()

                batch_feats = np.stack([b_mean, b_std, b_max], axis=1)
                fold_features.append(batch_feats)

                # Inference
                outputs = model(inputs)
                probs = torch.sigmoid(outputs).cpu().numpy()

                fold_preds.append(probs)
                fold_targets.append(targets.numpy())

        # Move model back to CPU to save memory
        model.cpu()

        fold_preds = np.concatenate(fold_preds)
        fold_targets = np.concatenate(fold_targets)
        fold_features = np.concatenate(fold_features)

        oof_preds.append(fold_preds)
        oof_targets.append(fold_targets)
        val_features_list.append(fold_features)

        fold_auc = calculate_roc_auc(fold_targets, fold_preds)
        print(f"Fold {fold_idx} Verified AUC: {fold_auc}")

    # 3. Overall Validation Metric
    all_oof_preds = np.concatenate(oof_preds)
    all_oof_targets = np.concatenate(oof_targets)
    all_val_features = np.concatenate(val_features_list)

    final_metric = calculate_roc_auc(all_oof_targets, all_oof_preds)
    print_metric("Final Validation Metric", final_metric)

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate error magnitude
    errors = np.abs(all_oof_targets.flatten() - all_oof_preds.flatten())

    feature_names = ["Spectrogram Mean", "Spectrogram Std", "Spectrogram Max"]

    print("Correlation between Error Magnitude and Input Features:")
    for i, name in enumerate(feature_names):
        feat_vals = all_val_features[:, i]

        # Check for constant values (std ~ 0) to avoid division by zero in correlation
        if np.std(feat_vals) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(errors, feat_vals)[0, 1]

        print(f"{name}: {corr:.4f}")

    # 5. Submission
    threshold = 0.9959177895986835
    if final_metric > threshold:
        print(
            f"\nMetric ({final_metric}) > Threshold ({threshold}). Generating Submission..."
        )

        test_loader = get_test_loader(load_cached_data=True)

        clip_prob_sum = {}

        # Ensemble Inference
        for i, model in enumerate(models):
            print(f"Inference with Model {i}...")
            model.to(Config.DEVICE)
            model.eval()

            with torch.no_grad():
                for inputs, clips in test_loader:
                    inputs = inputs.to(Config.DEVICE)
                    outputs = model(inputs)
                    probs = torch.sigmoid(outputs).cpu().numpy().flatten()

                    for clip, prob in zip(clips, probs):
                        if clip not in clip_prob_sum:
                            clip_prob_sum[clip] = 0.0
                        clip_prob_sum[clip] += prob

            model.cpu()

        # Average and Save
        final_clips = []
        final_probs = []

        for clip in clip_prob_sum:
            avg_prob = clip_prob_sum[clip] / Config.N_FOLDS
            final_clips.append(clip)
            final_probs.append(avg_prob)

        submission_df = pd.DataFrame({"clip": final_clips, "probability": final_probs})

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({threshold}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
