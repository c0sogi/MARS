import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd

# Import library components
from library.config import Config
from library.train import run_training
from library.inference import optimize_threshold, generate_submission, predict_with_tta
from library.dataset import get_dataloaders
from library.model import get_artwork_model
from library.utils import load_checkpoint, calculate_micro_f1, seed_everything


def main():
    # --- Configuration Overrides for Fast Baseline ---
    # We override the default configuration to ensure the script completes quickly (within 2 hours)
    # while still attempting to achieve a good score.
    # Training ResNet101d on ~96k images takes time; 12 epochs fits within the 4 hour limit.
    # Cite solution_lesson_node_00014: Enhancing Feature Preservation with ResNet-D Deep Stems
    Config.epochs = 12

    print(
        f"Configuration: Epochs={Config.epochs}, Batch Size={Config.batch_size}, Image Size={Config.img_size}"
    )

    # --- Training Phase ---
    print("\n=== Starting Training Phase ===")
    # run_training handles data loading, model setup, training loop, and saving the best model.
    best_val_score_train = run_training()
    print(
        f"Training completed. Best Validation Score (during training): {best_val_score_train}"
    )

    # --- Inference & Validation Phase ---
    print("\n=== Starting Inference & Validation Phase ===")
    device = Config.device

    # Load DataLoaders
    # We need the validation loader for metric calculation and failure analysis
    # We need the test loader for submission
    print("Loading dataloaders...")
    _, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Initialize Model
    model = get_artwork_model(num_classes=Config.num_classes, pretrained=False)
    model = model.to(device)

    # Load Best Checkpoint
    checkpoint_path = Config.model_save_path
    if os.path.exists(checkpoint_path):
        print(f"Loading best model from {checkpoint_path}...")
        load_checkpoint(checkpoint_path, model, device=device)
    else:
        print("WARNING: Model checkpoint not found! Using random weights.")

    # Optimize Threshold
    # This finds the best threshold on the validation set
    best_threshold = optimize_threshold(model, val_loader, device)

    # Generate Predictions on Validation Set for Final Metric & Failure Analysis
    print("Generating validation predictions with TTA...")
    val_probs, val_targets = predict_with_tta(model, val_loader, device)

    # Calculate Final Validation Metric
    # Note: calculate_micro_f1 expects probabilities and applies the threshold internally
    final_metric = calculate_micro_f1(val_probs, val_targets, threshold=best_threshold)
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    print("\n=== Starting Failure Analysis ===")
    # 1. Calculate Error Magnitude per Sample
    # We define error as the number of incorrect label predictions (Hamming distance)
    # Binarize predictions
    val_preds_bin = (val_probs.numpy() > best_threshold).astype(int)
    val_targets_np = val_targets.numpy()

    # Sum of absolute differences (0 if correct, 1 if FP or FN) per sample
    # Sum across classes (axis 1)
    sample_errors = np.sum(np.abs(val_preds_bin - val_targets_np), axis=1)

    # 2. Correlate with Input Features
    # We analyze a subset of the validation set to save time
    num_analysis_samples = min(1000, len(val_loader.dataset))
    print(f"Analyzing {num_analysis_samples} samples...")

    widths = []
    heights = []
    aspect_ratios = []
    mean_intensities = []
    subset_errors = []

    dataset = val_loader.dataset

    # Iterate through the first N samples
    for i in range(num_analysis_samples):
        try:
            # Get file path from dataframe
            row = dataset.df.iloc[i]
            # dataset.input_dir is defined in Config
            img_path = os.path.join(Config.input_dir, row["file_path"])

            # Read image to get properties
            img = cv2.imread(img_path)
            if img is not None:
                h, w, c = img.shape
                mean_val = np.mean(img)

                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h if h > 0 else 0)
                mean_intensities.append(mean_val)
                subset_errors.append(sample_errors[i])
        except Exception as e:
            continue

    # Calculate Correlations
    if len(subset_errors) > 1:
        # Use numpy for correlation to avoid extra dependencies
        corr_w = np.corrcoef(widths, subset_errors)[0, 1]
        corr_h = np.corrcoef(heights, subset_errors)[0, 1]
        corr_ar = np.corrcoef(aspect_ratios, subset_errors)[0, 1]
        corr_mean = np.corrcoef(mean_intensities, subset_errors)[0, 1]

        print(f"Correlation Error vs Width: {corr_w}")
        print(f"Correlation Error vs Height: {corr_h}")
        print(f"Correlation Error vs Aspect Ratio: {corr_ar}")
        print(f"Correlation Error vs Mean Intensity: {corr_mean}")
    else:
        print("Insufficient data for correlation analysis.")

    # --- Submission ---
    print("\n=== Submission Check ===")
    TARGET_THRESHOLD = 0.6311561264822134

    if final_metric > TARGET_THRESHOLD:
        print(
            f"Metric ({final_metric}) exceeds threshold ({TARGET_THRESHOLD}). Generating submission..."
        )
        generate_submission(model, test_loader, best_threshold, device)
    else:
        print(
            f"Metric ({final_metric}) does not exceed threshold ({TARGET_THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
