import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

# Import provided library modules
from library import config
from library import utils
from library import dataset
from library import model as model_lib
from library import engine


def main():
    # 1. Setup
    utils.seed_everything(config.SEED)
    device = config.DEVICE
    print(f"Running on device: {device}")

    # Storage for OOF analysis
    oof_preds = []
    oof_targets = []
    oof_angles = []
    oof_img_means = []

    # Storage for Test predictions (Accumulator for averaging)
    # We need to know the size of test set first, or we can append and stack later.
    # Since we iterate folds, we'll initialize a dictionary or array after the first fold.
    test_preds_accumulator = None
    test_ids_list = None

    # 2. Cross-Validation Loop
    for fold_idx in range(config.NUM_FOLDS):
        print(f"\n--- Starting Fold {fold_idx + 1}/{config.NUM_FOLDS} ---")

        # Get Data Loaders
        train_loader, val_loader = dataset.get_data_loaders(
            fold_idx, load_cached_data=True
        )

        # Initialize Model
        model = model_lib.SHH_SE_CNN().to(device)

        # Optimizer & Loss
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
        criterion = nn.BCEWithLogitsLoss()

        # Early Stopping
        early_stopping = engine.EarlyStopping(
            patience=config.PATIENCE, fold_idx=fold_idx
        )

        # Training Loop
        for epoch in range(config.NUM_EPOCHS):
            train_loss = engine.train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss = engine.evaluate(model, val_loader, criterion, device)

            # Check Early Stopping
            early_stopping(val_loss, model, optimizer, epoch)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        # Load Best Model for Inference
        best_model_path = os.path.join(
            config.CHECKPOINT_DIR, f"model_best_fold_{fold_idx}.pth"
        )
        checkpoint = utils.load_checkpoint(model, best_model_path)
        print(
            f"Loaded best model from epoch {checkpoint['epoch']} with loss {checkpoint['best_loss']:.4f}"
        )

        # --- Validation Inference (OOF) ---
        model.eval()
        fold_preds = []
        fold_targets = []
        fold_angles = []
        fold_means = []

        with torch.no_grad():
            for images, angles, targets in val_loader:
                images = images.to(device)
                angles_gpu = angles.to(device)

                # Forward
                logits = model(images, angles_gpu)
                probs = torch.sigmoid(logits)

                # Store results
                fold_preds.append(probs.cpu().numpy())
                fold_targets.append(targets.numpy())
                fold_angles.append(angles.numpy())

                # Calculate simple image stats for failure analysis (mean of first channel HH)
                # images is (B, 3, 75, 75)
                img_mean = images[:, 0, :, :].mean(dim=(1, 2)).cpu().numpy()
                fold_means.append(img_mean)

        oof_preds.append(np.concatenate(fold_preds))
        oof_targets.append(np.concatenate(fold_targets))
        oof_angles.append(np.concatenate(fold_angles))
        oof_img_means.append(np.concatenate(fold_means))

        # --- Test Inference ---
        test_loader = dataset.get_test_loader(load_cached_data=True)
        fold_test_preds = []
        fold_test_ids = []

        with torch.no_grad():
            for images, angles, ids in test_loader:
                images = images.to(device)
                angles_gpu = angles.to(device)

                logits = model(images, angles_gpu)
                probs = torch.sigmoid(logits)

                fold_test_preds.append(probs.cpu().numpy())
                fold_test_ids.extend(ids)

        fold_test_preds = np.concatenate(fold_test_preds)

        # Accumulate for Ensemble
        if test_preds_accumulator is None:
            test_preds_accumulator = fold_test_preds
            test_ids_list = fold_test_ids
        else:
            test_preds_accumulator += fold_test_preds

    # 3. Evaluation & Failure Analysis
    print("\n--- Evaluation & Failure Analysis ---")

    # Flatten OOF arrays
    all_oof_preds = np.concatenate(oof_preds)
    all_oof_targets = np.concatenate(oof_targets)
    all_oof_angles = np.concatenate(oof_angles)
    all_oof_means = np.concatenate(oof_img_means)

    # Calculate Metric
    final_metric = log_loss(all_oof_targets, all_oof_preds)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    # Calculate absolute error
    errors = np.abs(all_oof_preds - all_oof_targets)

    # Correlate error with features
    # Handle NaNs in angles just in case (though they should be imputed by loader)
    valid_mask = ~np.isnan(all_oof_angles)

    if np.sum(valid_mask) > 0:
        corr_angle, _ = pearsonr(errors[valid_mask], all_oof_angles[valid_mask])
        print(f"Correlation (Error vs Incidence Angle): {corr_angle:.4f}")
    else:
        print("Correlation (Error vs Incidence Angle): NaN (All angles missing)")

    corr_mean, _ = pearsonr(errors, all_oof_means)
    print(f"Correlation (Error vs Image Intensity): {corr_mean:.4f}")

    # 4. Submission
    THRESHOLD = 0.1806015565870406

    if final_metric < THRESHOLD:
        print("\nValidation metric meets threshold. Generating submission...")

        # Average test predictions
        avg_test_preds = test_preds_accumulator / config.NUM_FOLDS

        # Create DataFrame
        submission_df = pd.DataFrame(
            {"id": test_ids_list, "is_iceberg": avg_test_preds}
        )

        # Save
        submission_df.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")
    else:
        print(f"\nValidation metric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
