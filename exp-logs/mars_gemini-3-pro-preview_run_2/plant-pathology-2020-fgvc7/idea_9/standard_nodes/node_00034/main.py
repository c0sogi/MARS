import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib

# Import from provided libraries
from library.config import Config
from library.utils import seed_everything, calculate_roc_auc
from library.engine import run_fold
from library.stacking import StackingPipeline


def main():
    # 1. Setup
    seed_everything(Config.seed)

    # Adjust Config for Fast Baseline Execution
    # Reducing epochs to ensure completion within the time limit.
    # 6 epochs are sufficient for fine-tuning pre-trained models on this small dataset.
    Config.epochs = 6

    print("Configuration:")
    print(f"  Device: {Config.device}")
    print(f"  Epochs: {Config.epochs}")
    print(f"  Batch Size: {Config.batch_size}")
    print(f"  Working Dir: {Config.working_dir}")

    # 2. Train Base Models (Heterogeneous Ensemble)
    # We have 2 model architectures and 5 folds -> 10 training runs
    print("\n=== Starting Training Phase ===")
    for model_cfg in Config.models:
        for fold in range(Config.num_folds):
            # Train the model for the specific fold
            run_fold(fold, model_cfg)

    # 3. Stacking & Evaluation
    print("\n=== Starting Stacking Phase ===")
    pipeline = StackingPipeline()

    # Generate OOF Predictions (Features for Meta-Learner)
    # load_cached_data=False ensures we use the newly trained models from this run
    oof_df = pipeline.get_oof_predictions(load_cached_data=False)

    # Train Meta-Learner on OOF predictions
    meta_model = pipeline.train_meta_learner()

    # 4. Validation Assessment
    print("\n=== Validation Assessment ===")
    # Load the specific hold-out validation set metadata
    if not os.path.exists(Config.val_metadata_path):
        raise FileNotFoundError(
            f"Validation metadata not found at {Config.val_metadata_path}"
        )

    val_meta_df = pd.read_csv(Config.val_metadata_path)
    val_ids = set(val_meta_df["image_id"].values)

    # Filter OOF predictions to keep only the validation set samples
    val_preds_df = oof_df[oof_df["image_id"].isin(val_ids)].copy()

    if val_preds_df.empty:
        raise ValueError(
            "Validation predictions are empty. Check metadata and OOF alignment."
        )

    # Prepare features for Meta-Learner
    feature_cols = [c for c in val_preds_df.columns if c.startswith("model_")]
    X_val = val_preds_df[feature_cols].values

    # Get True Labels
    y_true = val_preds_df[["target_rust", "target_scab"]].values

    # Predict with Meta-Learner
    # predict_proba returns a list of [n_samples, 2] arrays for each target (MultiOutputClassifier)
    preds_proba = meta_model.predict_proba(X_val)
    y_pred_rust = preds_proba[0][:, 1]
    y_pred_scab = preds_proba[1][:, 1]
    y_pred = np.stack([y_pred_rust, y_pred_scab], axis=1)

    # Calculate Metric
    final_metric = calculate_roc_auc(y_true, y_pred)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Compute Error (Mean Absolute Error per sample averaged over classes)
    mae_per_sample = np.mean(np.abs(y_true - y_pred), axis=1)
    val_preds_df["error"] = mae_per_sample

    # Correlate with Metadata Features (File Size)
    file_sizes = []
    for img_id in val_preds_df["image_id"]:
        # Find path in val_meta_df
        rel_path = val_meta_df.loc[
            val_meta_df["image_id"] == img_id, "file_path"
        ].values[0]
        full_path = os.path.join(Config.input_dir, rel_path)
        if os.path.exists(full_path):
            file_sizes.append(os.path.getsize(full_path))
        else:
            file_sizes.append(0)

    val_preds_df["file_size"] = file_sizes

    # Calculate Correlation
    if val_preds_df["file_size"].std() > 0:
        corr = val_preds_df["error"].corr(val_preds_df["file_size"])
        print(f"Correlation between Model Error and Image File Size: {corr:.4f}")
    else:
        print("Could not calculate correlation (constant file size).")

    # 6. Submission
    print("\n=== Submission Generation ===")
    threshold = 0.9954104122251848
    if final_metric > threshold:
        print(
            f"Metric ({final_metric:.6f}) > Threshold ({threshold:.6f}). Generating submission..."
        )
        pipeline.generate_submission()
    else:
        print(
            f"Metric ({final_metric:.6f}) <= Threshold ({threshold:.6f}). Submission skipped."
        )


if __name__ == "__main__":
    main()
