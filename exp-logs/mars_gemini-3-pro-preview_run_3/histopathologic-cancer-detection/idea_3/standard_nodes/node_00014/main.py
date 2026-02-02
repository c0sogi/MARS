import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from scipy.stats import pearsonr

# Import from provided libraries
from library.config import Config
from library.utils import set_seed, compute_score
from library.dataset import prepare_folds, get_dataloaders, get_test_dataloader
from library.networks import get_model
from library.engine import train_fold, predict


def main():
    # --- 1. Setup & Configuration ---
    set_seed(Config.SEED)

    # Override Config for Full Convergence
    # 5 Folds * 1 Model * 20 Epochs fits comfortably within 4 hours on A100
    Config.EPOCHS = 20
    Config.BATCH_SIZE = 512

    print(
        f"Configuration: {Config.EPOCHS} Epochs, {Config.NUM_FOLDS} Folds, Models: {Config.MODEL_ARCHITECTURES}"
    )

    device = Config.DEVICE
    print(f"Using device: {device}")

    # Prepare data (folds)
    # This merges train and val metadata and creates stratified folds
    df_folds = prepare_folds(load_cached_data=True)

    # Initialize storage for OOF predictions
    # We will store predictions for the entire dataset to compute a global AUC
    oof_preds = np.zeros(len(df_folds))
    oof_targets = df_folds["label"].values

    # Store model paths for final inference
    model_paths = []

    # --- 2. Cross-Validation Loop ---
    for fold_id in range(Config.NUM_FOLDS):
        print(f"\n=== Starting Fold {fold_id}/{Config.NUM_FOLDS - 1} ===")

        # Get DataLoaders for this fold
        train_loader, val_loader = get_dataloaders(
            fold_id=fold_id,
            batch_size=Config.BATCH_SIZE,
        )

        # Indices for this validation fold
        val_indices = df_folds[df_folds["fold"] == fold_id].index.values

        fold_ensemble_preds = []

        for model_name in Config.MODEL_ARCHITECTURES:
            print(f"--- Training {model_name} on Fold {fold_id} ---")

            # Initialize Model
            model = get_model(model_name, pretrained=True)
            model = model.to(device)

            # Optimizer & Loss
            optimizer = optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=Config.EPOCHS, eta_min=1e-6
            )
            criterion = nn.BCEWithLogitsLoss()

            # Define Save Path
            save_name = f"{model_name}_fold_{fold_id}.pth"
            save_path = os.path.join(Config.WORKING_DIR, save_name)
            model_paths.append((model_name, save_path))

            # Train
            # train_fold handles the loop, validation, early stopping, and saving best model
            model = train_fold(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epochs=Config.EPOCHS,
                patience=Config.EARLY_STOPPING_PATIENCE,
                save_path=save_path,
                scheduler=scheduler,
            )

            # Generate predictions for this fold's validation set using TTA
            print(f"Generating predictions for {model_name} fold {fold_id}...")
            preds = predict(model, val_loader, device, use_tta=True)
            fold_ensemble_preds.append(preds)

            # Cleanup to save memory
            del model, optimizer, criterion
            torch.cuda.empty_cache()

        # Average predictions from both architectures for this fold
        avg_fold_preds = np.mean(fold_ensemble_preds, axis=0)

        # Store in global OOF array
        oof_preds[val_indices] = avg_fold_preds

    # --- 3. Validation & Failure Analysis ---
    print("\n=== Validation & Failure Analysis ===")

    # Compute Global OOF AUC
    final_val_metric = compute_score(oof_targets, oof_preds)
    print(f"Final Validation Metric: {final_val_metric}")

    # Failure Analysis
    # Calculate absolute error
    errors = np.abs(oof_targets - oof_preds)

    # Select a subset of validation data for feature analysis to save time
    # We use the indices from the dataframe
    analysis_indices = np.random.choice(
        len(df_folds), size=min(1000, len(df_folds)), replace=False
    )

    print("Computing feature correlations on validation subset...")
    brightness_values = []
    contrast_values = []
    subset_errors = errors[analysis_indices]

    for idx in analysis_indices:
        row = df_folds.iloc[idx]
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        try:
            img = cv2.imread(img_path)
            if img is None:
                brightness_values.append(0)
                contrast_values.append(0)
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            brightness_values.append(np.mean(img))
            contrast_values.append(np.std(img))
        except Exception:
            brightness_values.append(0)
            contrast_values.append(0)

    # Calculate correlations
    if len(subset_errors) > 1:
        corr_brightness, _ = pearsonr(subset_errors, brightness_values)
        corr_contrast, _ = pearsonr(subset_errors, contrast_values)

        print(f"Correlation between Error and Brightness: {corr_brightness:.4f}")
        print(f"Correlation between Error and Contrast: {corr_contrast:.4f}")

    # --- 4. Submission ---
    THRESHOLD = 0.9928749117769824

    if final_val_metric > THRESHOLD:
        print("\n=== Generating Submission ===")

        # Load Test Loader
        test_loader = get_test_dataloader(batch_size=Config.BATCH_SIZE)

        test_preds_accumulator = []

        # Iterate over all saved models
        for model_name, path in model_paths:
            print(f"Inference with {os.path.basename(path)}...")

            # Load Model Architecture
            model = get_model(model_name, pretrained=False)
            model.load_state_dict(torch.load(path, map_location=device))
            model = model.to(device)

            # Predict with TTA
            preds = predict(model, test_loader, device, use_tta=True)
            test_preds_accumulator.append(preds)

            del model
            torch.cuda.empty_cache()

        # Average all predictions
        final_test_preds = np.mean(test_preds_accumulator, axis=0)

        # Create Submission DataFrame
        df_test = pd.read_csv(Config.TEST_METADATA_PATH)
        submission_df = pd.DataFrame({"id": df_test["id"], "label": final_test_preds})

        # Save
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"Validation metric {final_val_metric} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
