import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

# Import library components
from library.config import Config
from library.utils import set_seed, do_kaggle_metric
from library.dataset import get_dataloaders
from library.trainer import SaltTrainer


def run_failure_analysis(trainer, val_loader, threshold):
    """
    Performs failure analysis by correlating model performance (AP) with metadata.
    """
    trainer.model.eval()
    device = trainer.device

    depths_all = []
    coverages_all = []
    scores_all = []

    # IoU thresholds for the metric
    iou_thresholds = np.linspace(0.5, 0.95, 10)

    with torch.no_grad():
        for images, masks, depths in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            depths_gpu = depths.to(device)

            # Forward pass
            logits = trainer.model(images, depths_gpu)
            probs = torch.sigmoid(logits)

            # Convert to numpy
            probs_np = probs.cpu().numpy()
            masks_np = masks.cpu().numpy()
            depths_np = depths.cpu().numpy()  # These are normalized depths

            # Un-normalize depths for interpretation if needed,
            # but correlation works fine on normalized data (linear transform).
            # We'll use the raw depths from the dataset if we want absolute values,
            # but the loader returns normalized ones.
            # To get raw depths, we can reverse the normalization: z = z_norm * std + mean
            # However, correlation coefficient is invariant to linear scaling.

            batch_size = images.size(0)

            # Binarize predictions
            preds_bin = (probs_np > threshold).astype(np.uint8)
            masks_bin = (masks_np > 0.5).astype(np.uint8)

            for i in range(batch_size):
                p = preds_bin[i, 0]
                t = masks_bin[i, 0]

                # Calculate Salt Coverage (Ground Truth)
                coverage = np.sum(t) / t.size
                coverages_all.append(coverage)

                # Store Depth (Normalized)
                depths_all.append(depths_np[i].item())

                # Calculate AP for this single image
                intersection = np.sum(p * t)
                union = np.sum(p) + np.sum(t) - intersection

                if union == 0:
                    iou = 1.0
                else:
                    iou = intersection / union

                # Average Precision over thresholds
                matches = iou > iou_thresholds
                avg_precision = np.mean(matches)
                scores_all.append(avg_precision)

    scores_all = np.array(scores_all)
    depths_all = np.array(depths_all)
    coverages_all = np.array(coverages_all)

    # Calculate Correlations
    # We correlate Error (1 - Score) or just Score.
    # Positive correlation with Score means "Higher X -> Better Model".
    # Negative correlation with Score means "Higher X -> Worse Model".

    corr_depth, _ = pearsonr(depths_all, scores_all)
    corr_cov, _ = pearsonr(coverages_all, scores_all)

    print(f"Correlation (Depth vs Performance): {corr_depth:.4f}")
    print(f"Correlation (Salt Coverage vs Performance): {corr_cov:.4f}")

    if corr_depth < -0.1:
        print("-> Model performs worse on deeper images.")
    elif corr_depth > 0.1:
        print("-> Model performs better on deeper images.")

    if corr_cov < -0.1:
        print("-> Model performs worse on images with more salt.")
    elif corr_cov > 0.1:
        print("-> Model performs better on images with more salt.")


def main():
    # 1. Setup and Configuration
    set_seed(Config.SEED)
    Config.setup()

    print(f"Running optimized training with {Config.EPOCHS} epochs...")

    # 2. Data Loading
    # Utilizing cached data for speed
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Training
    trainer = SaltTrainer()
    trainer.fit(train_loader, val_loader)

    # 4. Threshold Optimization
    print("\nOptimizing threshold...")
    best_threshold = trainer.optimize_threshold(val_loader)

    # 5. Final Validation Metric
    # Recalculate to ensure exact printing
    _, final_metric = trainer.validate(val_loader, threshold=best_threshold)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    run_failure_analysis(trainer, val_loader, best_threshold)

    # 7. Submission
    # Condition: Metric > 0.7893333333333333
    target_metric = 0.7893333333333333

    if final_metric > target_metric:
        print(
            f"\nMetric ({final_metric}) exceeds target ({target_metric}). Generating submission..."
        )
        trainer.predict(test_loader, threshold=best_threshold)
    else:
        print(
            f"\nMetric ({final_metric}) did not exceed target ({target_metric}). Skipping submission."
        )


if __name__ == "__main__":
    main()
