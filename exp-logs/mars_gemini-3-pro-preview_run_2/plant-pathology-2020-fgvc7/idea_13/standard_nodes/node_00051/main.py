import os
import cv2
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import (
    seed_everything,
    get_binary_targets,
    reconstruct_4_class_probabilities,
    calculate_metric,
    get_class_weights,
)
from library.data import get_folds_data
from library.training import run_fold
from library.inference import get_oof_predictions, get_test_predictions_raw
from library.stacking import run_stacking, generate_submission


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)

    # Optimize for fast baseline execution while ensuring convergence
    Config.EPOCHS = 10
    print(f"Configuration: EPOCHS={Config.EPOCHS}, DEVICE={Config.DEVICE}")

    # 2. Training Loop
    # Iterate over each model architecture and each fold
    for model_cfg in Config.MODELS:
        model_name = model_cfg["name"]
        for fold_idx in range(Config.N_FOLDS):
            # Check if checkpoint already exists to avoid re-training
            checkpoint_path = os.path.join(
                Config.CHECKPOINT_DIR, f"best_model_{model_name}_fold_{fold_idx}.pth"
            )

            if os.path.exists(checkpoint_path):
                print(
                    f"Checkpoint found for {model_name} fold {fold_idx}, skipping training."
                )
            else:
                print(f"Training {model_name} fold {fold_idx}...")
                run_fold(fold_idx, model_cfg)

    # 3. Inference (OOF and Test)
    # Generate OOF predictions for stacking and validation
    print("\nGenerating/Loading OOF Predictions...")
    oof_preds = get_oof_predictions(
        load_cached_data=True
    )  # Shape: (N_samples, N_models, 2)

    # Generate Raw Test predictions
    print("Generating/Loading Test Predictions...")
    test_preds_raw = get_test_predictions_raw(
        load_cached_data=True
    )  # Shape: (N_test, N_models, 2)

    # 4. Stacking and Validation
    print("\nRunning Stacking and Validation...")

    # Load fold data and targets
    df_folds = get_folds_data(load_cached_data=True)
    y_train_binary = get_binary_targets(df_folds)  # Shape: (N_samples, 2) [rust, scab]

    # We need to evaluate the "Stacked" model.
    # To get unbiased predictions of the stack on the training set, we use cross_val_predict
    # on the OOF predictions of the base models.

    stacked_oof_binary = np.zeros_like(y_train_binary)
    target_cols = Config.TARGET_COLS  # ["rust", "scab"]

    # Replicate the Rank-Based Stacking logic for validation
    for i, target in enumerate(target_cols):
        X_feat = oof_preds[:, :, i]  # (N_samples, N_models)
        y_target = y_train_binary[:, i]

        # Rank Normalization (Row-wise relative to dataset, but here column-wise)
        # The library implementation ranks column-wise (per model)
        X_ranked = np.apply_along_axis(
            lambda x: rankdata(x, method="average"), axis=0, arr=X_feat
        )
        X_ranked = X_ranked / X_ranked.shape[0]

        # Meta-Learner CV
        meta_learner = LogisticRegression(random_state=Config.SEED)

        # We use StratifiedKFold for the meta-learner CV to match the distribution
        # Note: We are stacking on OOFs, so simple CV here gives us "OOF of the Stack"
        cv = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )

        preds = cross_val_predict(
            meta_learner, X_ranked, y_target, cv=cv, method="predict_proba"
        )
        stacked_oof_binary[:, i] = preds[:, 1]

    # Reconstruct 4-class probabilities for the Stacked OOF
    stacked_oof_4class = reconstruct_4_class_probabilities(
        stacked_oof_binary[:, 0], stacked_oof_binary[:, 1]
    )

    # Load Hold-out Validation Metadata to identify the validation subset
    val_meta_df = pd.read_csv(Config.VAL_METADATA_PATH)
    val_ids = set(val_meta_df["image_id"].values)

    # Filter Stacked OOF predictions to only include the hold-out validation set
    is_val = df_folds["image_id"].isin(val_ids)
    val_preds = stacked_oof_4class[is_val]

    # Get Ground Truth for the validation set
    # We need the 4-class targets.
    # df_folds contains 'healthy', 'multiple_diseases', 'rust', 'scab' columns.
    val_targets_df = df_folds[is_val]
    val_targets = val_targets_df[Config.ORIGINAL_TARGET_COLS].values

    # Calculate Metric
    final_metric = calculate_metric(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate Mean Absolute Error per sample across the 4 classes
    # MAE is a good proxy for "error magnitude"
    errors = np.abs(val_targets - val_preds).mean(axis=1)

    # Collect metadata features for correlation
    file_sizes = []
    widths = []
    heights = []

    # We need to read image files to get dimensions if not in metadata
    # Optimizing: Read only validation images
    for idx, row in val_targets_df.iterrows():
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        try:
            # File size
            f_size = os.path.getsize(full_path)
            file_sizes.append(f_size)

            # Dimensions (Read header only if possible, but cv2 reads full)
            # Given small val set (328 images), reading is fast enough
            img = cv2.imread(full_path)
            if img is not None:
                h, w, _ = img.shape
                widths.append(w)
                heights.append(h)
            else:
                widths.append(0)
                heights.append(0)
        except Exception:
            file_sizes.append(0)
            widths.append(0)
            heights.append(0)

    # Calculate Correlations
    if len(errors) > 1:
        corr_size, _ = pearsonr(errors, file_sizes)
        corr_width, _ = pearsonr(errors, widths)
        corr_height, _ = pearsonr(errors, heights)

        print("Correlation between Error Magnitude and Features:")
        print(f"  File Size: {corr_size:.4f}")
        print(f"  Image Width: {corr_width:.4f}")
        print(f"  Image Height: {corr_height:.4f}")
    else:
        print("Insufficient data for correlation analysis.")

    # 6. Submission
    THRESHOLD = 0.9954104122251848
    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric:.6f}) > Threshold ({THRESHOLD:.6f}). Generating Submission..."
        )

        # Run full stacking pipeline (Train meta-learner on all OOF, predict Test)
        final_test_probs = run_stacking(
            oof_preds,
            test_preds_raw,
            y_train_binary,
            load_cached_data=False,  # Force re-run to ensure consistency
        )

        generate_submission(final_test_probs)
    else:
        print(
            f"\nMetric ({final_metric:.6f}) <= Threshold ({THRESHOLD:.6f}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
