import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from provided library files
from library.dataset import prepare_data, get_dataloaders
from library.engine import (
    run_regression_training,
    predict_depths,
    run_segmentation_training,
    predict_segmentation,
    DepthAwareLinkNet34,
    DEVICE,
)
from library.utils import calc_map, rle_encode, calc_iou

# Constants
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
THRESHOLD_SCORE = 0.7916666666666666


def set_seed(seed=42):
    """Sets fixed seeds for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def crop_center(img, target_shape=(101, 101)):
    """Crops the center of the image/mask to target shape."""
    if img.ndim == 3:
        h, w = img.shape[1], img.shape[2]
    else:
        h, w = img.shape[0], img.shape[1]

    th, tw = target_shape
    x1 = (w - tw) // 2
    y1 = (h - th) // 2

    if img.ndim == 3:
        return img[:, y1 : y1 + th, x1 : x1 + tw]
    else:
        return img[y1 : y1 + th, x1 : x1 + tw]


def get_val_predictions(model_path, loader):
    """
    Generates predictions and retrieves targets for the validation set.
    Crops predictions and targets back to the original 101x101 size.
    """
    model = DepthAwareLinkNet34(num_classes=1).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    all_probs = []
    all_targets = []
    all_depths = []

    with torch.no_grad():
        for batch in loader:
            images, masks, depths, _ = batch
            images = images.to(DEVICE)
            depths = depths.to(DEVICE)

            # Forward pass
            outputs = model(images, depths)
            probs = torch.sigmoid(outputs)

            # Convert to Numpy
            probs_np = probs.squeeze(1).cpu().numpy()
            masks_np = masks.squeeze(1).cpu().numpy()
            depths_np = depths.cpu().numpy().flatten()

            # Crop and store
            for p, m, d in zip(probs_np, masks_np, depths_np):
                p_crop = crop_center(p, (101, 101))
                m_crop = crop_center(m, (101, 101))

                all_probs.append(p_crop)
                all_targets.append(m_crop)
                all_depths.append(d)

    return np.array(all_probs), np.array(all_targets), np.array(all_depths)


def calculate_image_precision(pred_mask, target_mask):
    """Calculates Average Precision for a single image over IoU thresholds 0.5:0.95."""
    iou = calc_iou(pred_mask, target_mask)
    thresholds = np.arange(0.5, 0.96, 0.05)
    matches = iou > thresholds
    return np.mean(matches)


def main():
    set_seed(42)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    print("Initializing Data...")
    # Load data (cached if available)
    data_store = prepare_data(load_cached_data=True)
    train_loader, val_loader, test_loader = get_dataloaders(data_store, batch_size=32)

    # ---------------------------------------------------------
    # Stage 1: Depth Regression
    # ---------------------------------------------------------
    print("\n--- Stage 1: Depth Regression ---")
    # Train regressor to learn depth from image texture
    reg_model_path = run_regression_training(
        train_loader, val_loader, epochs=15, lr=1e-4, patience=5
    )

    print("Imputing Test Depths...")
    # Impute depths for the test set
    test_imputed_depths = predict_depths(reg_model_path, test_loader)

    # ---------------------------------------------------------
    # Stage 2: Segmentation
    # ---------------------------------------------------------
    print("\n--- Stage 2: Segmentation Training ---")
    # Train segmenter using ground truth depths
    seg_model_path = run_segmentation_training(
        train_loader, val_loader, epochs=30, lr=1e-3, patience=8
    )

    # ---------------------------------------------------------
    # Validation & Threshold Optimization
    # ---------------------------------------------------------
    print("\n--- Validation & Threshold Optimization ---")
    # Get raw probabilities for validation set
    val_probs, val_targets, val_depths = get_val_predictions(seg_model_path, val_loader)

    best_threshold = 0.5
    best_map = 0.0

    # Sweep thresholds to maximize mAP
    thresholds = np.arange(0.3, 0.75, 0.05)
    for t in thresholds:
        # Binarize predictions
        val_preds_bin = (val_probs > t).astype(np.uint8)
        # Calculate mAP
        score = calc_map(val_preds_bin, val_targets)
        if score > best_map:
            best_map = score
            best_threshold = t

    print(f"Best Threshold: {best_threshold:.2f}")
    print(f"Final Validation Metric: {best_map}")

    # ---------------------------------------------------------
    # Failure Analysis
    # ---------------------------------------------------------
    print("\n--- Failure Analysis ---")
    # Calculate per-image AP at best threshold
    val_preds_best = (val_probs > best_threshold).astype(np.uint8)
    image_aps = []
    salt_coverages = []

    for i in range(len(val_preds_best)):
        ap = calculate_image_precision(val_preds_best[i], val_targets[i])
        image_aps.append(ap)
        salt_coverages.append(np.mean(val_targets[i]))

    image_aps = np.array(image_aps)
    salt_coverages = np.array(salt_coverages)
    errors = 1.0 - image_aps

    # Calculate correlations
    # Handle potential constant arrays to avoid NaNs
    if np.std(errors) > 0 and np.std(val_depths) > 0:
        corr_depth = np.corrcoef(errors, val_depths)[0, 1]
    else:
        corr_depth = 0.0

    if np.std(errors) > 0 and np.std(salt_coverages) > 0:
        corr_salt = np.corrcoef(errors, salt_coverages)[0, 1]
    else:
        corr_salt = 0.0

    print(f"Correlation (Error vs Depth): {corr_depth:.4f}")
    print(f"Correlation (Error vs Salt Coverage): {corr_salt:.4f}")

    # ---------------------------------------------------------
    # Submission
    # ---------------------------------------------------------
    if best_map > THRESHOLD_SCORE:
        print("\n--- Generating Submission ---")
        # Predict Test set using imputed depths and TTA
        test_probs = predict_segmentation(
            seg_model_path, test_loader, imputed_depths=test_imputed_depths
        )

        # Binarize using optimized threshold
        test_preds = (test_probs > best_threshold).astype(np.uint8)

        # RLE Encode
        rles = []
        ids = data_store["test"]["ids"]

        for i in range(len(test_preds)):
            rle = rle_encode(test_preds[i])
            rles.append(rle)

        # Save to CSV
        sub_df = pd.DataFrame({"id": ids, "rle_mask": rles})
        sub_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation score {best_map} did not meet threshold {THRESHOLD_SCORE}. Skipping submission."
        )


if __name__ == "__main__":
    main()
