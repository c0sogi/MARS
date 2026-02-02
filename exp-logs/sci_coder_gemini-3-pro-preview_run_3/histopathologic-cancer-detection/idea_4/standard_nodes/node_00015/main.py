import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2

# Import from provided library
from library.config import Config
from library.utils import seed_everything, calculate_auc, get_device
from library.training import run_fold_training
from library.inference import generate_submission, predict_with_tta
from library.models import get_model
from library.data import get_dataloaders


def main():
    # 1. Setup
    print("Initializing Fast Baseline Run...")
    Config.setup()
    seed_everything(Config.SEED)
    device = get_device()

    # Configuration for Fast Baseline
    # We limit epochs to 2 to ensure completion within 2 hours on the full dataset.
    # Full dataset is required to attempt reaching the high AUC threshold.
    FAST_EPOCHS = 2

    # 2. Training Loop
    print(f"\n=== Starting Training (Epochs={FAST_EPOCHS}) ===")

    # Iterate over all architectures and folds
    for model_name in Config.MODEL_ARCHS:
        for fold_id in range(Config.NUM_FOLDS):
            print(f"\n--- Training {model_name} | Fold {fold_id} ---")
            run_fold_training(
                fold_id=fold_id,
                model_name=model_name,
                num_epochs=FAST_EPOCHS,
                patience=FAST_EPOCHS,  # Disable early stopping for short run
                load_cached_data=True,
            )

    # 3. Validation & Metric Calculation
    print("\n=== Performing OOF Validation ===")

    # We need to generate OOF predictions for the specific validation set (metadata/val.csv).
    # Since the library mixes train/val for CV, we identify the fold assignment for each val image
    # and use the model trained on the other folds (i.e., the model for that fold) to predict.

    # Load the hold-out validation metadata
    df_val_holdout = pd.read_csv(Config.VAL_META_PATH)
    holdout_ids = set(df_val_holdout["id"].values)

    # Container for OOF predictions: {id: [pred_arch1, pred_arch2, ...]}
    oof_preds_accum = {}
    oof_labels = {}

    # Iterate through folds to generate predictions
    for fold_id in range(Config.NUM_FOLDS):
        print(f"Generating predictions for Fold {fold_id} validation set...")

        # Get dataloader for this fold's validation set
        # Note: We don't need the train loader, so we ignore it
        _, val_loader = get_dataloaders(fold_id=fold_id, load_cached_data=True)

        # Get IDs for this fold
        fold_df = val_loader.dataset.df
        fold_ids = fold_df["id"].values

        # Initialize predictions for this fold
        fold_preds_ensemble = np.zeros(len(fold_ids))

        # Predict with each architecture
        for model_name in Config.MODEL_ARCHS:
            safe_model_name = model_name.split(".")[0]
            weight_path = os.path.join(
                Config.WORK_DIR, f"{safe_model_name}_fold_{fold_id}.pth"
            )

            # Load model
            model = get_model(model_name, pretrained=False)
            model.load_state_dict(torch.load(weight_path, map_location=device))
            model = model.to(device)

            # Predict
            preds = predict_with_tta(model, val_loader, device)
            fold_preds_ensemble += preds

            del model
            torch.cuda.empty_cache()

        # Average across architectures
        fold_preds_ensemble /= len(Config.MODEL_ARCHS)

        # Store predictions for IDs that are in the hold-out set
        # (The CV splits contain both train.csv and val.csv data)
        for idx, img_id in enumerate(fold_ids):
            if img_id in holdout_ids:
                oof_preds_accum[img_id] = fold_preds_ensemble[idx]
                # Store label (ground truth)
                # val_loader dataset returns (image, label), but we can get label from df
                oof_labels[img_id] = fold_df.iloc[idx]["label"]

    # Align predictions with the hold-out dataframe order
    final_preds = []
    final_targets = []

    # Filter to ensure we only evaluate on the requested hold-out set
    valid_ids = []
    for img_id in df_val_holdout["id"].values:
        if img_id in oof_preds_accum:
            final_preds.append(oof_preds_accum[img_id])
            final_targets.append(oof_labels[img_id])
            valid_ids.append(img_id)
        else:
            # This should not happen if folds cover all data
            print(f"Warning: ID {img_id} not found in OOF predictions.")

    final_preds = np.array(final_preds)
    final_targets = np.array(final_targets)

    # Calculate Metric
    val_auc = calculate_auc(final_targets, final_preds)
    print(f"Final Validation Metric: {val_auc}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")

    # Calculate errors
    errors = np.abs(final_targets - final_preds)

    # Select a subset for image analysis to save time (e.g., top 1000 errors + 1000 random)
    # Actually, let's just do 2000 random samples to get a distribution
    n_samples = min(2000, len(valid_ids))
    indices = np.random.choice(len(valid_ids), n_samples, replace=False)

    brightness_vals = []
    contrast_vals = []
    sampled_errors = []

    print(f"Analyzing {n_samples} samples for feature correlation...")

    for idx in indices:
        img_id = valid_ids[idx]
        # Find path
        row = df_val_holdout[df_val_holdout["id"] == img_id].iloc[0]
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        try:
            img = cv2.imread(full_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Simple features
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            brightness = np.mean(gray)
            contrast = np.std(gray)

            brightness_vals.append(brightness)
            contrast_vals.append(contrast)
            sampled_errors.append(errors[idx])

        except Exception:
            continue

    # Calculate Correlations
    if len(sampled_errors) > 1:
        corr_bright = np.corrcoef(sampled_errors, brightness_vals)[0, 1]
        corr_contrast = np.corrcoef(sampled_errors, contrast_vals)[0, 1]

        print(f"Correlation (Error vs Brightness): {corr_bright:.4f}")
        print(f"Correlation (Error vs Contrast): {corr_contrast:.4f}")
    else:
        print("Could not compute correlations (no samples).")

    # 5. Submission
    THRESHOLD = 0.9933607469455475
    if val_auc > THRESHOLD:
        print(f"\nValidation metric {val_auc} > {THRESHOLD}. Generating submission...")
        generate_submission(load_cached_data=True)
    else:
        print(f"\nValidation metric {val_auc} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
