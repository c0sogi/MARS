import os
import sys
import numpy as np
import pandas as pd
import torch

from library.config import Config
from library.utils import set_seed, get_logger, do_kaggle_metric
from library.dataset import get_dataloaders
from library.model import ResNet34WideLinkNet
from library.trainer import Trainer
from library.inference import InferenceEngine


def calculate_per_image_ap(predict_probs, truth_masks, threshold=0.5):
    """
    Calculates Average Precision for each image individually.
    Replicates the logic of do_kaggle_metric but returns the array of scores.
    """
    # Binarize
    predict = (predict_probs > threshold).astype(np.uint8)
    truth = truth_masks.astype(np.uint8)

    ious = []
    # Calculate IoU for each image
    for p, t in zip(predict, truth):
        p_flat = p.flatten()
        t_flat = t.flatten()

        intersection = np.sum(p_flat * t_flat)
        union = np.sum(p_flat) + np.sum(t_flat) - intersection

        if union == 0:
            iou = 1.0
        else:
            iou = intersection / union
        ious.append(iou)

    ious = np.array(ious)

    # Thresholds: 0.5, 0.55, ..., 0.95
    iou_thresholds = np.arange(0.5, 1.0, 0.05)

    # Compare IoUs to thresholds -> Shape: (N_images, N_thresholds)
    hits = ious[:, None] > iou_thresholds[None, :]

    # Average precision per image (mean over thresholds)
    ap_per_image = np.mean(hits, axis=1)

    return ap_per_image


def main():
    # 1. Setup
    # Override epochs for fast baseline execution as requested
    Config.EPOCHS = 15
    set_seed(Config.SEED)
    logger = get_logger(Config.WORKING_DIR)
    device = torch.device(Config.DEVICE)

    logger.info("Initializing Fast Baseline Pipeline...")

    # 2. Data Loading
    logger.info("Loading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders()

    # 3. Model Initialization
    logger.info(f"Initializing model: {Config.MODEL_NAME}")
    model = ResNet34WideLinkNet()
    model.to(device)

    # 4. Training
    logger.info("Starting training...")
    trainer = Trainer(model, train_loader, val_loader, device)
    trainer.fit()

    # 5. Load Best Model for Inference
    best_ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_ckpt_path):
        logger.info(f"Loading best model from {best_ckpt_path}")
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    else:
        logger.warning("Best model checkpoint not found! Using current model state.")

    # 6. Validation & Threshold Optimization
    inference_engine = InferenceEngine(model, device)

    # Get raw probabilities and masks from validation set
    val_probs, val_masks = inference_engine.predict_val(val_loader)

    # Optimize threshold
    best_threshold = 0.5
    best_score = -1.0

    # Sweep thresholds to find best mAP
    thresholds = np.arange(0.3, 0.76, 0.05)
    for t in thresholds:
        score = do_kaggle_metric(val_probs, val_masks, threshold=t)
        if score > best_score:
            best_score = score
            best_threshold = t

    logger.info(f"Optimal Threshold: {best_threshold:.4f} with mAP: {best_score:.10f}")

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {best_score:.10f}")

    # 7. Failure Analysis
    logger.info("Performing Failure Analysis...")

    # Calculate per-image AP at the best threshold
    ap_per_image = calculate_per_image_ap(
        val_probs, val_masks, threshold=best_threshold
    )
    error_magnitude = 1.0 - ap_per_image

    # Load validation metadata to correlate with features
    val_df = pd.read_csv(Config.VAL_CSV)

    # Ensure alignment (val_loader is sequential, non-shuffled)
    if len(val_df) != len(error_magnitude):
        logger.warning(
            f"Metadata size {len(val_df)} != Predictions size {len(error_magnitude)}. Alignment might be off."
        )

    # Extract features
    depths = val_df["z"].values
    salt_coverage = val_df["salt_coverage"].values

    # Calculate correlations
    # Handle NaNs in depth if any (though dataset info says 0 nan)
    valid_indices = ~np.isnan(depths)

    corr_depth = np.corrcoef(depths[valid_indices], error_magnitude[valid_indices])[
        0, 1
    ]
    corr_coverage = np.corrcoef(salt_coverage, error_magnitude)[0, 1]

    print("-" * 30)
    print("FAILURE ANALYSIS REPORT")
    print("-" * 30)
    print(f"Correlation (Error vs Depth): {corr_depth:.4f}")
    print(f"Correlation (Error vs Salt Coverage): {corr_coverage:.4f}")
    print("-" * 30)

    # 8. Submission
    SUBMISSION_THRESHOLD = 0.7985

    if best_score > SUBMISSION_THRESHOLD:
        logger.info(
            f"Validation score {best_score:.4f} exceeds threshold {SUBMISSION_THRESHOLD}. Generating submission..."
        )
        inference_engine.generate_submission(test_loader, threshold=best_threshold)
    else:
        logger.info(
            f"Validation score {best_score:.4f} does not exceed threshold {SUBMISSION_THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
