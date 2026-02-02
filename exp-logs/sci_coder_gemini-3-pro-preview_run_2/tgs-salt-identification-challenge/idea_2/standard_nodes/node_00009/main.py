import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Import library modules
from library.train import run_training
from library.inference import predict_and_submit
from library.model import DepthConditionedLinkNet
from library.dataset import SaltDataset, get_transforms
from library.utils import set_seed

# Constants
METADATA_VAL = "./metadata/val.csv"
MODEL_PATH = "./working/idea_2/best_model.pth"
SUBMISSION_PATH = "./submission/submission.csv"
THRESHOLD_SCORE = 0.7893333333333333
BATCH_SIZE = 32
NUM_WORKERS = 4


def calculate_iou_vectorized(preds, labels):
    """
    Calculates IoU for a batch of binary masks.
    preds: (N, H, W) boolean array
    labels: (N, H, W) boolean array
    Returns: (N,) float array of IoU values
    """
    intersection = (preds & labels).sum(axis=(1, 2))
    union = (preds | labels).sum(axis=(1, 2))

    iou = np.ones_like(intersection, dtype=np.float64)
    # If union > 0, calc iou. If union == 0 (both empty), iou = 1.
    mask = union > 0
    iou[mask] = intersection[mask] / union[mask]
    return iou


def compute_map_iou(iou_values):
    """
    Computes the mean Average Precision at different IoU thresholds (0.5 to 0.95).
    """
    # Thresholds 0.5 to 0.95 step 0.05
    thresholds = np.arange(0.5, 0.95 + 1e-6, 0.05)
    # For each iou, how many thresholds does it pass?
    # (N, 1) > (1, 10) -> (N, 10)
    passed = iou_values[:, None] > thresholds[None, :]
    # Mean over thresholds for each image -> Average Precision per image
    return passed.mean(axis=1)


def main():
    # Set seed for reproducibility
    set_seed(42)

    # 1. Training Phase
    print("Starting Training Phase...")
    # Train for 50 epochs as planned
    run_training(
        epochs=50,
        batch_size=BATCH_SIZE,
        lr=1e-3,
        num_workers=NUM_WORKERS,
        patience=10,
        load_cached_data=True,
    )

    # 2. Validation & Metric Calculation Phase
    print("Starting Validation Phase...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    model = DepthConditionedLinkNet(num_classes=1)
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Load Validation Data
    # We use 'val' mode but 'val' transforms (which only has padding/norm, no aug)
    val_dataset = SaltDataset(
        mode="val",
        metadata_path=METADATA_VAL,
        load_cached_data=True,
        transform=get_transforms("val"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )

    # Metadata for failure analysis
    df_val = pd.read_csv(METADATA_VAL)
    id_to_coverage = dict(zip(df_val["id"], df_val["salt_coverage"]))
    id_to_depth = dict(zip(df_val["id"], df_val["z"]))

    all_ious = []
    all_depths = []
    all_coverages = []

    # Crop parameters (must match inference.py logic to reverse padding)
    crop_top = 13
    crop_bottom = 13 + 101
    crop_left = 13
    crop_right = 13 + 101

    with torch.no_grad():
        for images, masks, depths, ids in val_loader:
            images = images.to(device)
            depths_gpu = depths.to(device)

            # --- TTA Inference ---
            # 1. Original
            outputs_orig = model(images, depths_gpu)
            preds_orig = torch.sigmoid(outputs_orig)

            # 2. Flip
            images_flip = torch.flip(images, dims=[3])
            outputs_flip = model(images_flip, depths_gpu)
            preds_flip_raw = torch.sigmoid(outputs_flip)
            preds_flip = torch.flip(preds_flip_raw, dims=[3])

            # Average
            preds_avg = (preds_orig + preds_flip) / 2.0

            # --- Post-Processing ---
            preds_np = preds_avg.cpu().numpy()[:, 0, :, :]  # (B, 128, 128)
            masks_np = masks.numpy()  # (B, 128, 128) - dataset returns padded masks

            # Crop to original 101x101
            preds_cropped = preds_np[:, crop_top:crop_bottom, crop_left:crop_right]
            masks_cropped = masks_np[:, crop_top:crop_bottom, crop_left:crop_right]

            # Binarize
            preds_bin = preds_cropped > 0.5
            masks_bin = (
                masks_cropped > 0.5
            )  # Masks are 0 or 1 float/int, convert to bool

            # Calculate IoU
            batch_ious = calculate_iou_vectorized(preds_bin, masks_bin)
            all_ious.extend(batch_ious)

            # Collect metadata
            for img_id in ids:
                all_depths.append(id_to_depth[img_id])
                all_coverages.append(id_to_coverage[img_id])

    all_ious = np.array(all_ious)
    scores = compute_map_iou(all_ious)
    final_metric = scores.mean()

    print(f"Final Validation Metric: {final_metric}")

    # 3. Failure Analysis
    print("Performing Failure Analysis...")
    errors = 1.0 - scores

    # Correlation with Depth
    if len(all_depths) > 1:
        corr_depth = np.corrcoef(errors, all_depths)[0, 1]
    else:
        corr_depth = 0.0

    # Correlation with Salt Coverage
    if len(all_coverages) > 1:
        corr_cov = np.corrcoef(errors, all_coverages)[0, 1]
    else:
        corr_cov = 0.0

    print(f"Correlation between Error and Depth: {corr_depth}")
    print(f"Correlation between Error and Salt Coverage: {corr_cov}")

    # 4. Submission
    if final_metric > THRESHOLD_SCORE:
        print(f"Metric {final_metric} > {THRESHOLD_SCORE}. Generating submission...")
        predict_and_submit(
            model_path=MODEL_PATH,
            output_path=SUBMISSION_PATH,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            load_cached_data=True,
        )
    else:
        print(f"Metric {final_metric} <= {THRESHOLD_SCORE}. Submission skipped.")


if __name__ == "__main__":
    main()
