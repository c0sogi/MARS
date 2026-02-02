import os
import sys
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

# 1. Configuration Patching for Fast Baseline
# We modify the configuration before importing dependent modules to ensure
# the training runs quickly (reduced epochs) while maintaining the logic.
import library.config

library.config.NUM_EPOCHS = 10  # Reduce epochs for speed (default 15)

# 2. Imports
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    WORKING_DIR,
    SEED,
    DEVICE,
    BATCH_SIZE,
    NUM_WORKERS,
    USE_AMP,
    NUM_FOLDS,
)
from library.train_eval import run_training, predict_and_submit
from library.dataset import RNWIVDataset, get_transforms
from library.model import RNWIVEfficientNet
from library.data_processing import process_dataset_roi


def main():
    # Set seeds for reproducibility in the main script as well
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ==========================================
    # 1. Training Phase
    # ==========================================
    print("Starting RN-WIV Training Pipeline...")
    # This function handles the 5-fold CV training using the patched config
    run_training()

    # ==========================================
    # 2. Validation & Failure Analysis
    # ==========================================
    print("\nStarting Validation and Failure Analysis...")

    # Load Metadata
    if not os.path.exists(TRAIN_METADATA_PATH) or not os.path.exists(VAL_METADATA_PATH):
        print("Error: Metadata not found.")
        return

    df_train = pd.read_csv(TRAIN_METADATA_PATH)
    df_val = pd.read_csv(VAL_METADATA_PATH)

    # Combine to recreate the exact CV splits used in training
    df_full = pd.concat([df_train, df_val], ignore_index=True)

    # Ensure ROI data is available (should be cached by training step)
    roi_df = process_dataset_roi(df_full, load_cached_data=True)

    # Recreate Splits
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    oof_preds = np.zeros(len(df_full))
    oof_targets = df_full["MGMT_value"].values

    # Generate OOF Predictions
    # We iterate through the folds, load the trained model, and predict on the validation set
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df_full, df_full["MGMT_value"])
    ):
        model_path = os.path.join(WORKING_DIR, f"best_model_fold{fold}.pth")

        if not os.path.exists(model_path):
            print(f"Warning: Model for fold {fold} not found. Skipping.")
            continue

        # Prepare Validation Data for this fold
        val_sub = df_full.iloc[val_idx].reset_index(drop=True)
        val_ds = RNWIVDataset(
            val_sub, roi_df, transform=get_transforms("val"), is_train=False
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        # Load Model
        model = RNWIVEfficientNet().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()

        fold_preds = []

        # Inference
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(DEVICE)
                with torch.amp.autocast("cuda", enabled=USE_AMP):
                    logits = model(images)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    fold_preds.extend(probs)

        # Store predictions
        oof_preds[val_idx] = np.array(fold_preds).flatten()

    # Compute Final Metric
    final_auc = roc_auc_score(oof_targets, oof_preds)
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis
    print("\n=== Failure Analysis ===")
    df_full["pred"] = oof_preds
    df_full["error"] = np.abs(df_full["MGMT_value"] - df_full["pred"])

    # Merge with ROI stats to analyze correlation
    # roi_df contains columns like 'flair_count', 'flair_start', etc.
    analysis_df = df_full.merge(roi_df, on="BraTS21ID", how="left")

    # Identify numerical features for correlation
    feature_cols = [
        c
        for c in analysis_df.columns
        if any(x in c for x in ["count", "start", "end", "size"])
    ]
    # Filter out non-numeric just in case
    feature_cols = [
        c for c in feature_cols if pd.api.types.is_numeric_dtype(analysis_df[c])
    ]

    print("Correlation between Error Magnitude and Input Features:")
    for col in feature_cols:
        if analysis_df[col].nunique() > 1:
            # Handle potential NaNs by filling with 0
            vec_feature = analysis_df[col].fillna(0).values
            vec_error = analysis_df["error"].values

            # Compute Pearson correlation using numpy
            if np.std(vec_feature) > 0 and np.std(vec_error) > 0:
                corr = np.corrcoef(vec_error, vec_feature)[0, 1]
                print(f"{col}: {corr:.4f}")
            else:
                print(f"{col}: NaN (Constant values)")

    # ==========================================
    # 3. Submission
    # ==========================================
    THRESHOLD = 0.6705454545454544

    if final_auc > THRESHOLD:
        print(
            f"\nValidation metric {final_auc} exceeds threshold {THRESHOLD}. Generating submission..."
        )
        predict_and_submit()
    else:
        print(
            f"\nValidation metric {final_auc} does not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
