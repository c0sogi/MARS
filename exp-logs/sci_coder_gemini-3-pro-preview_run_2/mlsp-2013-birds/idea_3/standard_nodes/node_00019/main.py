import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import cv2
from scipy.stats import pearsonr
from skmultilearn.model_selection import IterativeStratification
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import seed_everything, calculate_roc_auc, load_checkpoint
from library.dataset import load_data, BirdDataset, get_transforms
from library.model import BirdClassifier
from library.engine import validate_one_epoch
from library.train import run_kfold_training


def main():
    # 1. Setup & Configuration
    seed_everything(Config.SEED)

    # Override Config for Fast Baseline
    # Limit epochs to ensure execution within time limits while allowing convergence
    Config.EPOCHS = 5
    # Batch size appropriate for A100
    Config.BATCH_SIZE = 32

    # Redirect submission output temporarily to control final release based on metric
    TEMP_SUB_DIR = os.path.join(Config.WORKING_DIR, "temp_submission")
    REAL_SUB_DIR = Config.SUBMISSION_DIR
    Config.SUBMISSION_DIR = TEMP_SUB_DIR
    os.makedirs(TEMP_SUB_DIR, exist_ok=True)

    print("Starting Fast Baseline Pipeline...")

    # 2. Run Training
    # This trains the models across folds and generates a submission in TEMP_SUB_DIR
    # We disable debug mode to use the full dataset (it's small enough)
    run_kfold_training(debug=False)

    # 3. OOF Validation & Metric Calculation
    print("\nPerforming OOF Validation...")

    # Load data exactly as the training script does (merged train + val)
    df_train, _ = load_data()

    # Prepare for Stratification (Replicating library.train logic to identify fold splits)
    # We need dummy X and labels y for IterativeStratification
    X_dummy = np.zeros((len(df_train), 1))
    label_cols = [f"species_{i}" for i in range(Config.NUM_CLASSES)]
    y_labels = df_train[label_cols].values

    splitter = IterativeStratification(n_splits=Config.N_FOLDS, order=1)

    # Arrays to store Out-Of-Fold predictions and targets
    oof_preds = np.zeros(y_labels.shape)
    oof_targets = np.zeros(y_labels.shape)

    # Iterate folds to generate OOF predictions
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X_dummy, y_labels)):
        # Get validation subset for this fold
        df_fold_val = df_train.iloc[val_idx].reset_index(drop=True)

        # Setup Dataset & Loader
        val_dataset = BirdDataset(df_fold_val, transforms=get_transforms(data="valid"))
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        # Load Model
        model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.pth")
        if not os.path.exists(model_path):
            print(f"Model for fold {fold} not found. Skipping/Filling with zeros.")
            continue

        model = BirdClassifier(
            backbone=Config.BACKBONE, pretrained=False, num_classes=Config.NUM_CLASSES
        )
        model.to(Config.DEVICE)

        # Load checkpoint
        load_checkpoint(model_path, model, device=Config.DEVICE)

        # Inference
        # validate_one_epoch returns (loss, preds, targets)
        _, preds, targets = validate_one_epoch(model, val_loader, Config.DEVICE)

        # Assign predictions to the corresponding indices in the OOF arrays
        oof_preds[val_idx] = preds
        oof_targets[val_idx] = targets

        # Cleanup
        del model
        torch.cuda.empty_cache()

    # Calculate Final Metric (ROC AUC)
    final_metric = calculate_roc_auc(oof_targets, oof_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis on Validation Set...")

    # Calculate error per sample (Mean Absolute Error across classes)
    # We use MAE as a proxy for "error magnitude"
    errors = np.abs(oof_targets - oof_preds)
    mean_errors = np.mean(errors, axis=1)  # Shape: (N_samples,)

    # Extract Metadata Features for Correlation
    # Feature 1: Label Cardinality (Number of birds present)
    cardinality = np.sum(oof_targets, axis=1)

    # Feature 2 & 3: Image Statistics (Mean, Std)
    # We need to load images to compute these stats.
    pixel_means = []
    pixel_stds = []

    for idx, row in df_train.iterrows():
        # Reconstruct path logic from Dataset class
        orig_path = row["file_path_spec"]
        filename = os.path.basename(orig_path)
        full_path = os.path.join(Config.IMAGE_DIR, filename)

        # Load image (grayscale for stats)
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Handle missing files gracefully
            pixel_means.append(0)
            pixel_stds.append(0)
        else:
            pixel_means.append(np.mean(img))
            pixel_stds.append(np.std(img))

    pixel_means = np.array(pixel_means)
    pixel_stds = np.array(pixel_stds)

    # Calculate Pearson Correlations
    # We check if error correlates with complexity (cardinality) or signal strength (pixel stats)
    corr_card, _ = pearsonr(mean_errors, cardinality)
    corr_mean, _ = pearsonr(mean_errors, pixel_means)
    corr_std, _ = pearsonr(mean_errors, pixel_stds)

    print("Correlation between Error Magnitude and Input Features:")
    print(f"  Label Cardinality: {corr_card:.4f}")
    print(f"  Pixel Intensity Mean: {corr_mean:.4f}")
    print(f"  Pixel Intensity Std: {corr_std:.4f}")

    # 5. Submission Logic
    # Threshold defined in task
    THRESHOLD = 0.8739452549958209

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating Submission..."
        )
        os.makedirs(REAL_SUB_DIR, exist_ok=True)
        src_file = os.path.join(TEMP_SUB_DIR, "submission.csv")
        dst_file = os.path.join(REAL_SUB_DIR, "submission.csv")

        if os.path.exists(src_file):
            shutil.copy(src_file, dst_file)
            print(f"Submission saved to {dst_file}")
        else:
            print("Error: Temporary submission file not found.")
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
