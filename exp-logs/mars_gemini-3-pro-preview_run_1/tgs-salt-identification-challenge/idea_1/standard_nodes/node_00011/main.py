import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import warnings

# Import from provided library files
from library.utils import set_seed
from library.dataset import get_dataloaders
from library.model import DepthAwareUNet
from library.trainer import SaltTrainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def calculate_image_precision(pred_mask, true_mask):
    """
    Calculates the average precision over IoU thresholds (0.5 to 0.95, step 0.05)
    for a single image pair.
    """
    # Check if masks are empty
    pred_empty = np.sum(pred_mask) == 0
    true_empty = np.sum(true_mask) == 0

    # Both empty: Perfect match
    if pred_empty and true_empty:
        return 1.0

    # One empty, one not: Complete failure
    if pred_empty or true_empty:
        return 0.0

    # Both non-empty: Calculate IoU
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    iou = intersection / union

    # Calculate score over thresholds
    thresholds = np.arange(0.5, 1.0, 0.05)
    matches = iou > thresholds
    return np.mean(matches)


def main():
    # 1. Setup
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Data Loading
    # Using cached data to speed up loading
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=32, num_workers=2, load_cached_data=True
    )

    # 3. Model Initialization
    # Input channels: 2 (Grayscale Image + Depth Map)
    model = DepthAwareUNet(n_channels=2, n_classes=1, bilinear=False)
    model.to(device)

    # 4. Training
    # Initialize trainer
    trainer = SaltTrainer(
        model=model,
        device=device,
        learning_rate=1e-3,
        checkpoint_dir="./working/idea_1",
    )

    # Train with extended epochs and scheduler (Cite solution_lesson_node_00004)
    # Increased epochs to 80 and patience to 25 to allow for late-stage boundary refinement (Cite solution_lesson_node_00010)
    trainer.train(train_loader, val_loader, epochs=80, patience=25)

    # 5. Validation Assessment and Failure Analysis
    print("\n--- Starting Validation Assessment ---")
    trainer.load_best_model()
    model.eval()

    precisions = []
    errors = []
    depths = []
    img_means = []
    img_stds = []

    # Padding parameters for cropping back to original size during validation check
    # Target 128x128 -> Original 101x101
    crop_top = 13
    crop_left = 13
    orig_h = 101
    orig_w = 101

    with torch.no_grad():
        for inputs, masks, _ in val_loader:
            inputs = inputs.to(device)
            masks = masks.to(device)

            # Forward pass with TTA (Cite solution_lesson_node_00004)
            logits = model(inputs)
            probs = torch.sigmoid(logits)

            inputs_flipped = torch.flip(inputs, dims=[3])
            logits_flipped = model(inputs_flipped)
            probs_flipped = torch.sigmoid(logits_flipped)
            probs_flipped_back = torch.flip(probs_flipped, dims=[3])

            probs = (probs + probs_flipped_back) / 2.0

            # Convert to numpy
            inputs_np = inputs.cpu().numpy()
            probs_np = probs.cpu().numpy()
            masks_np = masks.cpu().numpy()

            batch_size = inputs.shape[0]

            for i in range(batch_size):
                # Extract prediction and ground truth
                # Crop back to 101x101 to match competition metric logic strictly
                # (though IoU on padded vs unpadded is similar, strict metric usually implies original size)
                prob_map = probs_np[i, 0]
                true_mask = masks_np[i, 0]

                # Crop
                prob_crop = prob_map[
                    crop_top : crop_top + orig_h, crop_left : crop_left + orig_w
                ]
                true_crop = true_mask[
                    crop_top : crop_top + orig_h, crop_left : crop_left + orig_w
                ]

                # Binarize prediction
                pred_binary = (prob_crop > 0.5).astype(np.uint8)
                true_binary = (true_crop > 0.5).astype(np.uint8)

                # Calculate Metric
                score = calculate_image_precision(pred_binary, true_binary)
                precisions.append(score)

                # Failure Analysis Data Collection
                error = 1.0 - score
                errors.append(error)

                # Extract features from input tensor
                # Channel 0: Image, Channel 1: Depth Map
                img_channel = inputs_np[i, 0]
                depth_channel = inputs_np[i, 1]

                # Depth is constant across the map, take mean
                depth_val = np.mean(depth_channel)

                # Image stats (on the padded image is fine for correlation)
                img_mean = np.mean(img_channel)
                img_std = np.std(img_channel)

                depths.append(depth_val)
                img_means.append(img_mean)
                img_stds.append(img_std)

    # Calculate Final Metric
    final_metric = np.mean(precisions)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis Correlations
    print("\n--- Failure Analysis ---")
    if len(errors) > 1:
        corr_depth, _ = pearsonr(errors, depths)
        corr_mean, _ = pearsonr(errors, img_means)
        corr_std, _ = pearsonr(errors, img_stds)

        print(f"Correlation (Error vs Depth): {corr_depth:.4f}")
        print(f"Correlation (Error vs Image Mean): {corr_mean:.4f}")
        print(f"Correlation (Error vs Image Std): {corr_std:.4f}")
    else:
        print("Insufficient data for correlation analysis.")

    # 6. Submission
    threshold_metric = 0.7603333333333334
    if final_metric > threshold_metric:
        print(
            f"\n--- Generating Submission (Metric {final_metric:.4f} > {threshold_metric:.4f}) ---"
        )
        submission_path = "./submission/submission.csv"
        trainer.generate_submission(test_loader, output_file=submission_path)
    else:
        print(
            f"\n--- Skipping Submission (Metric {final_metric:.4f} <= {threshold_metric:.4f}) ---"
        )


if __name__ == "__main__":
    main()
