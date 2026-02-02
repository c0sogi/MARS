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
from library.train import train_single_model, predict_test


def load_model_for_inference(model_path, device):
    """
    Loads a model state dict.
    """
    model = IcebergResNet18(pretrained=False)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_with_tta(model, loader, device):
    """
    Performs inference with TTA (Original + Horizontal Flip).
    """
    preds = []
    with torch.no_grad():
        for images, angles, _ in loader:
            images = images.to(device)
            angles = angles.to(device)

            # TTA: Original
            out1 = model(images, angles)
            prob1 = torch.sigmoid(out1)

            # TTA: Horizontal Flip
            images_flip = torch.flip(images, [3])
            out2 = model(images_flip, angles)
            prob2 = torch.sigmoid(out2)

            # Average
            avg_prob = (prob1 + prob2) / 2.0
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

    # 3. Single Split Training (Cite solution_lesson_node_00020)
    print("Loading metadata for split...")
    df_train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    df_val_meta = pd.read_csv(Config.VAL_META_PATH)

    train_idx = df_train_meta["sample_index"].values
    val_idx = df_val_meta["sample_index"].values

    print(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}")

    # Train Single Model
    model_path = train_single_model(train_idx, val_idx, data, Config.WORKING_DIR)
    saved_model_paths = [model_path]

    # --- Validation Inference ---
    print(f"Generating predictions for Validation Set...")

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
    oof_preds = np.zeros(len(y_train_all))
    val_preds = predict_with_tta(model, val_loader, device)

    # Fill oof_preds only for validation indices for metric calculation
    # We only care about the validation subset metric
    oof_preds_subset = val_preds

    # 4. Global Evaluation
    # We calculate metric only on the validation set
    final_metric = log_loss(y_val, oof_preds_subset)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate Error Magnitude on Validation Set
    errors = np.abs(y_val - oof_preds_subset)

    # Extract Features for correlation (Validation Set Only)
    feat_angle = ang_val
    feat_img_mean = np.mean(X_val[:, :, :, 0], axis=(1, 2))
    feat_img_std = np.std(X_val[:, :, :, 0], axis=(1, 2))

    # Calculate Correlations
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
