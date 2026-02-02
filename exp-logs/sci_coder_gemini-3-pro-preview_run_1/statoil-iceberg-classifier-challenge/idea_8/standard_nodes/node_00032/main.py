import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

# Ensure library is in path
sys.path.append(os.getcwd())

# Import from library
from library.config import Config
from library.utils import seed_everything
from library.data import process_and_cache_data, IcebergDataset, get_transforms
from library.model import IcebergResNet18
from library.train import run_fold, predict_test


def load_model_for_inference(model_path, device):
    """
    Loads a model state dict, handling potential key mismatches from SWA/DataParallel.
    """
    model = IcebergResNet18(pretrained=False)
    state_dict = torch.load(model_path, map_location=device)

    # Clean state_dict keys
    if "n_averaged" in state_dict:
        del state_dict["n_averaged"]

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model


def predict_with_tta(model, loader, device):
    """
    Performs inference with TTA (Original + H-Flip + V-Flip).
    """
    preds = []
    with torch.no_grad():
        for images, angles, _ in loader:
            images = images.to(device)
            angles = angles.to(device)

            # TTA 1: Original
            out1 = model(images, angles)
            prob1 = torch.sigmoid(out1)

            # TTA 2: Horizontal Flip
            images_h = torch.flip(images, [3])
            out2 = model(images_h, angles)
            prob2 = torch.sigmoid(out2)

            # TTA 3: Vertical Flip
            images_v = torch.flip(images, [2])
            out3 = model(images_v, angles)
            prob3 = torch.sigmoid(out3)

            # Average
            avg_prob = (prob1 + prob2 + prob3) / 3.0
            preds.extend(avg_prob.cpu().numpy().flatten())

    return np.array(preds)


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # 2. Data Loading
    print("Loading data...")
    data = process_and_cache_data(load_cached_data=True)

    X_train_all = data["train_images"]
    ang_train_all = data["train_angles"]
    y_train_all = data["train_labels"]

    # 3. Stratified K-Fold Training
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    oof_preds = np.zeros(len(y_train_all))
    saved_model_paths = []

    print(f"Starting {Config.N_FOLDS}-Fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_all, y_train_all)):
        # Run training for this fold
        # run_fold handles the training loop, SWA, and saving the best model
        model_path = run_fold(fold, train_idx, val_idx, data, Config.WORKING_DIR)
        saved_model_paths.append(model_path)

        # --- Validation Inference for OOF ---
        print(f"Generating OOF predictions for Fold {fold}...")

        # Prepare Validation Loader
        X_val = X_train_all[val_idx]
        ang_val = ang_train_all[val_idx]
        y_val = y_train_all[val_idx]

        val_dataset = IcebergDataset(
            X_val, ang_val, y_val, transform=get_transforms("valid")
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Load Model
        model = load_model_for_inference(model_path, device)

        # Predict
        fold_preds = predict_with_tta(model, val_loader, device)
        oof_preds[val_idx] = fold_preds

        # Fold Metric
        fold_loss = log_loss(y_val, fold_preds)
        print(f"Fold {fold} Log Loss: {fold_loss:.6f}")

    # 4. Global Evaluation
    final_metric = log_loss(y_train_all, oof_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate Error Magnitude
    errors = np.abs(y_train_all - oof_preds)

    # Extract Features for correlation
    # 1. Incidence Angle (use original raw values if possible, but normalized is fine for correlation magnitude)
    # We use the normalized angles from data['train_angles']
    feat_angle = ang_train_all

    # 2. Image Stats (Mean and Std of the first channel - HH)
    # X_train_all is (N, 224, 224, 3) after processing.
    # Note: process_and_cache_data scales images. We use these scaled values.
    feat_img_mean = np.mean(X_train_all[:, :, :, 0], axis=(1, 2))
    feat_img_std = np.std(X_train_all[:, :, :, 0], axis=(1, 2))

    # Calculate Correlations
    # Handle NaN in angles if any (though they should be imputed by now)
    valid_mask = ~np.isnan(feat_angle)

    corr_angle, _ = pearsonr(feat_angle[valid_mask], errors[valid_mask])
    corr_mean, _ = pearsonr(feat_img_mean, errors)
    corr_std, _ = pearsonr(feat_img_std, errors)

    print("Correlation between Error Magnitude and Input Features:")
    print(f"Incidence Angle: {corr_angle:.4f}")
    print(f"Image Mean (Band 1): {corr_mean:.4f}")
    print(f"Image Std (Band 1): {corr_std:.4f}")

    # 6. Submission Logic
    THRESHOLD = 0.17822679498532543

    if final_metric < THRESHOLD:
        print(
            f"\nValidation metric ({final_metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Generate predictions on test set using the ensemble
        ids, preds = predict_test(saved_model_paths, data, device)

        # Save submission
        df_sub = pd.DataFrame({"id": ids, "is_iceberg": preds})
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation metric ({final_metric}) did NOT meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
