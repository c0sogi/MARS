import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr

from library.config import Config
from library.dataset import get_dataloaders
from library.model import SaltNet
from library.utils import do_kaggle_metric, unpad_image, rle_encode
from library.train import train_model, set_seed


def get_validation_metrics(model, loader, device):
    """
    Runs inference on validation set and returns raw probabilities and masks
    for threshold optimization and failure analysis.
    """
    model.eval()
    all_probs = []
    all_masks = []
    all_ids = []
    all_depths = []

    with torch.no_grad():
        for images, masks, depths, ids in loader:
            images = images.to(device)
            depths_gpu = depths.to(device)

            # Inference
            logits = model(images, depths_gpu)
            probs = torch.sigmoid(logits).cpu().numpy()
            true_masks = masks.numpy()

            # Unpad
            for i in range(len(probs)):
                p = unpad_image(probs[i, 0], Config.ORIG_SIZE)
                m = unpad_image(true_masks[i, 0], Config.ORIG_SIZE)

                all_probs.append(p)
                all_masks.append(m)
                all_ids.append(ids[i])
                all_depths.append(depths[i].item())

    return (
        np.array(all_probs),
        np.array(all_masks),
        np.array(all_ids),
        np.array(all_depths),
    )


def calculate_image_precision(predict_mask, truth_mask):
    """
    Calculates the average precision for a single image across IoU thresholds 0.5-0.95.
    """
    intersection = np.sum(predict_mask * truth_mask)
    union = np.sum(predict_mask) + np.sum(truth_mask) - intersection

    if union == 0:
        iou = 1.0
    else:
        iou = intersection / union

    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    score = (iou > iou_thresholds).mean()
    return score


def generate_submission(model, loader, device, threshold, output_path):
    """
    Generates submission file with TTA and optimized threshold.
    """
    model.eval()
    predictions = {}

    print(f"Generating submission with threshold {threshold:.4f}...")

    with torch.no_grad():
        for images, depths, ids in loader:
            images = images.to(device)
            depths = depths.to(device)

            # TTA: Original
            logits = model(images, depths)
            probs = torch.sigmoid(logits)

            # TTA: Horizontal Flip
            if Config.TTA_FLIP:
                images_flip = torch.flip(images, dims=[3])
                logits_flip = model(images_flip, depths)
                probs_flip = torch.sigmoid(logits_flip)
                probs_flip = torch.flip(probs_flip, dims=[3])
                probs = (probs + probs_flip) / 2.0

            probs = probs.cpu().numpy()

            for i, img_id in enumerate(ids):
                # Unpad
                pred_mask = unpad_image(probs[i, 0], Config.ORIG_SIZE)

                # Binarize with optimized threshold
                pred_bin = (pred_mask > threshold).astype(np.uint8)

                # RLE Encode
                rle = rle_encode(pred_bin)
                predictions[img_id] = rle

    sub_df = pd.DataFrame.from_dict(predictions, orient="index", columns=["rle_mask"])
    sub_df.index.name = "id"
    sub_df.reset_index(inplace=True)
    sub_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    Config.create_dirs()
    device = torch.device(Config.DEVICE)

    # 2. Train
    print("Starting Training...")
    # We use the full dataset and epochs defined in Config to ensure high performance
    train_model(Config)

    # 3. Load Best Model
    print("\nLoading best model for validation and analysis...")
    model = SaltNet().to(device)
    if not os.path.exists(Config.CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {Config.CHECKPOINT_PATH}")

    model.load_state_dict(torch.load(Config.CHECKPOINT_PATH, map_location=device))
    model.eval()

    # 4. Validation & Threshold Optimization
    _, val_loader, test_loader = get_dataloaders(Config)

    print("Running inference on validation set...")
    val_probs, val_masks, val_ids, val_depths = get_validation_metrics(
        model, val_loader, device
    )

    print("Optimizing threshold...")
    thresholds = np.linspace(0.3, 0.7, 17)  # 0.30, 0.325, ..., 0.70
    best_threshold = 0.5
    best_score = 0.0

    for t in thresholds:
        score = do_kaggle_metric(val_probs, val_masks, threshold=t)
        if score > best_score:
            best_score = score
            best_threshold = t

    print(f"Best Threshold: {best_threshold:.4f}")
    print(f"Final Validation Metric: {best_score:.16f}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate per-image scores at best threshold
    per_image_scores = []
    binarized_preds = (val_probs > best_threshold).astype(np.uint8)

    for i in range(len(val_probs)):
        s = calculate_image_precision(binarized_preds[i], val_masks[i])
        per_image_scores.append(s)

    per_image_scores = np.array(per_image_scores)

    # Calculate error (1 - score)
    errors = 1.0 - per_image_scores

    # Load metadata to get coverage info (depths are already in val_depths)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    # Ensure alignment by ID
    val_meta.set_index("id", inplace=True)
    val_coverages = val_meta.loc[val_ids]["salt_coverage"].values

    # Correlations
    # We correlate Error with Depth and Salt Coverage
    corr_depth, _ = pearsonr(errors, val_depths)
    corr_cov, _ = pearsonr(errors, val_coverages)

    print(f"Correlation (Error vs Depth): {corr_depth:.4f}")
    print(f"Correlation (Error vs Salt Coverage): {corr_cov:.4f}")

    if abs(corr_depth) > 0.1:
        print("-> Model performance is sensitive to depth.")
    if abs(corr_cov) > 0.1:
        print("-> Model performance is sensitive to salt coverage size.")

    # 6. Submission
    target_metric = 0.7896666666666667
    if best_score > target_metric:
        print(
            f"\nValidation metric ({best_score:.4f}) meets target ({target_metric:.4f}). Generating submission..."
        )
        generate_submission(
            model, test_loader, device, best_threshold, Config.SUBMISSION_PATH
        )
    else:
        print(
            f"\nValidation metric ({best_score:.4f}) did not meet target ({target_metric:.4f}). Submission skipped."
        )


if __name__ == "__main__":
    main()
