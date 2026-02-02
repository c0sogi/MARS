import os
import sys
import pandas as pd
import numpy as np
import soundfile as sf
from sklearn.metrics import roc_auc_score
import pickle

# Import Library Modules
from library.config import Config
from library.utils import seed_everything
from library.runner import run_fold
from library.ensemble import (
    generate_pseudo_labels,
    generate_oof_features,
    train_meta_learner,
    predict_submission,
)


def analyze_failures(val_df, y_true, y_pred):
    """
    Performs failure analysis by correlating error with audio features.
    """
    print("Performing Failure Analysis...")
    errors = np.abs(y_true - y_pred)

    durations = []
    rms_values = []
    peaks = []

    # Iterate over validation samples to extract features
    for idx, row in val_df.iterrows():
        file_path = os.path.join(Config.input_root, row["file_path"])
        try:
            # Fast metadata read
            info = sf.info(file_path)
            dur = info.duration

            # Read audio for signal stats
            data, sr = sf.read(file_path)
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            rms = np.sqrt(np.mean(data**2))
            peak = np.max(np.abs(data))

            durations.append(dur)
            rms_values.append(rms)
            peaks.append(peak)
        except Exception as e:
            # Fallback for any read errors
            durations.append(0)
            rms_values.append(0)
            peaks.append(0)

    analysis_df = pd.DataFrame(
        {"error": errors, "duration": durations, "rms": rms_values, "peak": peaks}
    )

    print("Correlation between Model Error and Input Features:")
    print(f"Duration: {analysis_df['duration'].corr(analysis_df['error']):.4f}")
    print(f"RMS: {analysis_df['rms'].corr(analysis_df['error']):.4f}")
    print(f"Peak: {analysis_df['peak'].corr(analysis_df['error']):.4f}")


def main():
    # --- Configuration Override for Fast Baseline ---
    # We override specific Config attributes to ensure the run completes within the time limit
    # while still verifying the full pipeline logic on the complete dataset.
    Config.n_folds = 2  # Reduce folds from 5 to 2
    Config.epochs = 1  # Reduce epochs from 20 to 1
    Config.debug = False  # Use full dataset to ensure meaningful metrics

    # Ensure working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    # Set global seed
    seed_everything(Config.seed)

    print(f"Starting Pipeline with n_folds={Config.n_folds}, epochs={Config.epochs}")

    # ==========================================
    # ROUND 1: Initial Supervised Training
    # ==========================================
    print("\n=== Round 1: Initial Training ===")
    for fold in range(Config.n_folds):
        for model_name in Config.model_names:
            # Train and save checkpoints (Best AUC and Best Loss)
            run_fold(fold_idx=fold, model_name=model_name, load_cached_data=True)

    # ==========================================
    # Pseudo-Labeling (Consensus-Based)
    # ==========================================
    print("\n=== Generating Pseudo-Labels ===")
    checkpoint_dir = os.path.join(Config.working_dir, "checkpoints")

    # Generate expanded training set using Round 1 models
    # This creates 'pseudo_train.csv' in working_dir
    pseudo_train_csv = generate_pseudo_labels(checkpoint_dir, load_cached_data=False)

    # ==========================================
    # ROUND 2: Self-Distillation
    # ==========================================
    print("\n=== Round 2: Self-Distillation ===")

    # Update Config to point to the expanded dataset
    Config.train_csv = pseudo_train_csv

    # Update cache filename to prevent loading Round 1 cache
    # The pseudo-labeled dataset is larger, so we need a new cache.
    Config.train_cache_file = os.path.join(
        Config.working_dir, "cached_train_mels_round2.npy"
    )

    for fold in range(Config.n_folds):
        for model_name in Config.model_names:
            # Retrain models from scratch on expanded data
            # Overwrites Round 1 checkpoints (desired for final ensemble)
            run_fold(fold_idx=fold, model_name=model_name, load_cached_data=False)

    # ==========================================
    # Meta-Learning (Stacking)
    # ==========================================
    print("\n=== Training Meta-Learner ===")

    # Generate OOF features from Round 2 models
    # Note: Validation set remains the original pure validation set (handled by get_dataloaders)
    X_oof, y_oof, sorted_keys = generate_oof_features(
        checkpoint_dir, load_cached_data=True
    )

    # Train Logistic Regression
    meta_learner_path = os.path.join(Config.working_dir, "meta_learner.pkl")
    clf = train_meta_learner(X_oof, y_oof, meta_learner_path)

    # ==========================================
    # Validation & Failure Analysis
    # ==========================================
    print("\n=== Validation & Failure Analysis ===")

    # Predict on OOF features to get final ensemble validation probabilities
    oof_probs = clf.predict_proba(X_oof)[:, 1]

    # Calculate Final Metric
    final_auc = roc_auc_score(y_oof, oof_probs)
    print(f"Final Validation Metric: {final_auc}")

    # Load validation metadata for file paths
    val_df = pd.read_csv(Config.val_csv)

    # Perform Failure Analysis
    analyze_failures(val_df, y_oof, oof_probs)

    # ==========================================
    # Submission
    # ==========================================
    submission_threshold = 0.9998881660199745

    if final_auc > submission_threshold:
        print(
            f"\nMetric ({final_auc}) > Threshold ({submission_threshold}). Generating Submission..."
        )
        submission_path = "./submission/submission.csv"

        # Generate final predictions using Round 2 models + Meta-Learner
        predict_submission(
            round2_checkpoint_dir=checkpoint_dir,
            meta_learner_path=meta_learner_path,
            output_path=submission_path,
            load_cached_data=True,
        )
    else:
        print(
            f"\nMetric ({final_auc}) <= Threshold ({submission_threshold}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
