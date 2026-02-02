import os
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr

from library.config import Config, seed_everything
from library.trainer import train_fold
from library.inference import generate_submission, predict_with_tta
from library.models import MultiLevelEfficientNet, SwinTransformerModel
from library.dataset import AppleDataset, get_transforms
from library.utils import calculate_metric


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # 2. Load Metadata
    # We treat the provided 'train.csv' as our development set for Cross-Validation
    # We treat the provided 'val.csv' as the hold-out set for final evaluation
    train_dev_df = pd.read_csv(Config.TRAIN_METADATA)
    val_holdout_df = pd.read_csv(Config.VAL_METADATA)

    # 3. 5-Fold Stratified Cross-Validation Training
    # We split the development set into 5 folds
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Use 'stratify_label' for stratification
    y_strat = train_dev_df["stratify_label"]

    print("Starting 5-Fold Cross-Validation Training...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_dev_df, y_strat)):
        print(f"\n--- Fold {fold} ---")

        # Create fold-specific dataframes
        fold_train_df = train_dev_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = train_dev_df.iloc[val_idx].reset_index(drop=True)

        # Train EfficientNet-B4
        # Note: train_fold handles model initialization, training, and saving best checkpoint
        train_fold(fold, "effnet", fold_train_df, fold_val_df)

        # Train Swin Transformer
        train_fold(fold, "swin", fold_train_df, fold_val_df)

    # 4. Final Evaluation on Hold-out Validation Set
    print("\n=== Final Evaluation on Hold-out Validation Set ===")

    device = Config.DEVICE
    architectures = [("effnet", Config.IMG_SIZE_EFFNET), ("swin", Config.IMG_SIZE_SWIN)]

    # Initialize ensemble probability accumulator
    ensemble_probs = np.zeros(
        (len(val_holdout_df), Config.NUM_CLASSES), dtype=np.float64
    )
    model_count = 0

    # Iterate through all trained models
    for model_type, img_size in architectures:
        # Prepare DataLoader for hold-out set (labeled=False to match inference signature)
        transforms = get_transforms("valid", img_size)
        # We pass labeled=False because predict_with_tta expects just images
        val_ds = AppleDataset(
            val_holdout_df, Config.INPUT_DIR, transform=transforms, labeled=False
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        for fold in range(Config.N_FOLDS):
            ckpt_path = os.path.join(
                Config.WORKING_DIR, f"{model_type}_fold_{fold}_best.pth"
            )

            if not os.path.exists(ckpt_path):
                print(f"Warning: Checkpoint {ckpt_path} not found. Skipping.")
                continue

            # Initialize model
            if model_type == "effnet":
                model = MultiLevelEfficientNet(pretrained=False)
            else:
                model = SwinTransformerModel(pretrained=False)

            # Load weights
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.to(device)

            # Predict
            preds = predict_with_tta(model, val_loader, device)
            ensemble_probs += preds
            model_count += 1

            # Cleanup
            del model
            torch.cuda.empty_cache()

    if model_count == 0:
        raise RuntimeError("No models trained. Cannot evaluate.")

    # Average predictions
    final_probs = ensemble_probs / model_count

    # Calculate Metric
    y_true = val_holdout_df[Config.LABEL_COLS].values
    metric = calculate_metric(y_true, final_probs)

    print(f"Final Validation Metric: {metric}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")

    # Calculate error magnitude: 1 - probability assigned to the true class
    # y_true is one-hot, so sum(probs * y_true) gives prob of correct class
    prob_correct = np.sum(final_probs * y_true, axis=1)
    error_magnitude = 1.0 - prob_correct

    # Extract image features (Brightness and Contrast)
    brightness_vals = []
    contrast_vals = []

    for rel_path in val_holdout_df["file_path"]:
        full_path = os.path.join(Config.INPUT_DIR, rel_path)
        img = cv2.imread(full_path)
        if img is None:
            # Fallback for safety, though paths are validated
            brightness_vals.append(127.0)
            contrast_vals.append(50.0)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness_vals.append(np.mean(gray))
        contrast_vals.append(np.std(gray))

    # Calculate Correlations
    if len(error_magnitude) > 1:
        corr_bright, _ = pearsonr(error_magnitude, brightness_vals)
        corr_contrast, _ = pearsonr(error_magnitude, contrast_vals)
    else:
        corr_bright, corr_contrast = 0.0, 0.0

    print(f"Correlation (Error vs Brightness): {corr_bright:.4f}")
    print(f"Correlation (Error vs Contrast): {corr_contrast:.4f}")

    # 6. Submission
    # Threshold from requirements
    THRESHOLD = 0.9897675030297739

    if metric > THRESHOLD:
        print(
            f"\nMetric ({metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission()
    else:
        print(f"\nMetric ({metric}) <= Threshold ({THRESHOLD}). Skipping submission.")


if __name__ == "__main__":
    main()
