import sys
import os
import numpy as np
import pandas as pd
import torch
import cv2
from library.config import Config, seed_everything
from library.train import train_specific_model
from library.inference import (
    get_model_predictions,
    ensemble_predictions,
    generate_submission,
)
from library.utils import optimize_threshold
from library.dataset import get_dataloaders


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print("=== Starting Orchestration Script ===")

    # 2. Train Models
    # We train for 7 epochs to maximize performance within the 4-hour limit (Cite Lesson 00003).
    # Using the full dataset (debug=False) is crucial for reaching the target score.
    print(f"\n[Training] Starting training for Model A: {Config.MODEL_A_NAME}")
    train_specific_model(Config.MODEL_A_NAME, epochs=Config.EPOCHS, debug=False)

    print(f"\n[Training] Starting training for Model B: {Config.MODEL_B_NAME}")
    train_specific_model(Config.MODEL_B_NAME, epochs=Config.EPOCHS, debug=False)

    # 3. Validation & Metric Calculation
    print("\n[Validation] Loading validation data...")
    # We only need the validation loader here to get targets and run inference
    _, val_loader, _ = get_dataloaders(debug=False, load_cached_data=True)

    print("[Validation] Generating ensemble predictions...")
    # Get predictions for Model A
    probs_a, targets = get_model_predictions(
        Config.MODEL_A_NAME, "val", val_loader, device, load_cached_data=True
    )
    # Get predictions for Model B
    probs_b, _ = get_model_predictions(
        Config.MODEL_B_NAME, "val", val_loader, device, load_cached_data=True
    )

    # Ensemble predictions (simple average)
    ensemble_probs = ensemble_predictions([probs_a, probs_b])

    # Ensure targets are numpy array
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    elif isinstance(targets, list):
        targets = np.array(targets)

    # Optimize threshold
    best_thresh, best_score = optimize_threshold(targets, ensemble_probs)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {best_score}")

    # 4. Failure Analysis
    print("\n[Analysis] Performing Failure Analysis...")

    # Calculate per-sample F1 to determine error magnitude
    # Binarize predictions using the optimal threshold
    preds_bin = (ensemble_probs > best_thresh).astype(int)

    # Calculate F1 per sample (instance-level)
    # F1 = 2*TP / (2*TP + FP + FN)
    tp = np.sum((preds_bin == 1) & (targets == 1), axis=1)
    fp = np.sum((preds_bin == 1) & (targets == 0), axis=1)
    fn = np.sum((preds_bin == 0) & (targets == 1), axis=1)

    epsilon = 1e-7
    f1_samples = (2 * tp) / (2 * tp + fp + fn + epsilon)
    error_magnitude = 1.0 - f1_samples

    # Load metadata to get features
    val_df = pd.read_csv(Config.VAL_CSV)

    # Feature 1: Label Cardinality (from ground truth)
    # Handle potential NaNs in attribute_ids
    val_df["attribute_ids"] = val_df["attribute_ids"].fillna("")
    val_df["num_labels"] = val_df["attribute_ids"].apply(
        lambda x: len(x.split()) if x.strip() else 0
    )

    # Feature 2: Image Brightness (from image content)
    # We process a subset of validation images to save time
    sample_size = min(2000, len(val_df))
    sample_indices = np.random.choice(len(val_df), size=sample_size, replace=False)

    brightness_values = []
    sampled_errors = []
    sampled_cardinality = []

    print(f"  Processing {sample_size} images for feature extraction...")
    for idx in sample_indices:
        row = val_df.iloc[idx]
        path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Read image
        img = cv2.imread(path)
        if img is not None:
            # Calculate mean brightness
            b = np.mean(img) / 255.0
            brightness_values.append(b)
            sampled_errors.append(error_magnitude[idx])
            sampled_cardinality.append(val_df.iloc[idx]["num_labels"])

    # Calculate Correlations
    if len(sampled_errors) > 1:
        # Correlation with Label Cardinality
        corr_card = np.corrcoef(sampled_cardinality, sampled_errors)[0, 1]
        print(f"Correlation (Error vs Label Cardinality): {corr_card:.4f}")

        # Correlation with Brightness
        corr_bright = np.corrcoef(brightness_values, sampled_errors)[0, 1]
        print(f"Correlation (Error vs Image Brightness): {corr_bright:.4f}")
    else:
        print("Insufficient data for correlation analysis.")

    # 5. Submission
    THRESHOLD_SCORE = 0.6561

    print(f"\n[Submission] Checking threshold: {best_score} > {THRESHOLD_SCORE}")

    if best_score > THRESHOLD_SCORE:
        print("Threshold met. Generating submission file...")
        # generate_submission handles test inference and saving
        # It will reuse the cached validation probabilities we generated earlier
        generate_submission(debug=False, load_cached_data=True)
    else:
        print("Threshold not met. Skipping submission generation.")

    print("\n=== Workflow Completed ===")


if __name__ == "__main__":
    main()
