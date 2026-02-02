import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
from scipy.spatial.distance import directed_hausdorff
from torch.utils.data import DataLoader
import warnings

# Import library modules
from library.config import Config
from library.utils import set_seed
from library.dataset import UWMadisonDataset
from library.model import DeepLabV3Plus
from library.train import run_training
from library.inference import run_inference, remove_small_objects_3d

# Suppress warnings
warnings.filterwarnings("ignore")


def compute_hausdorff_3d(mask_pred, mask_gt, spacing_norm):
    """
    Computes 3D Hausdorff distance between two binary masks.
    Coordinates are normalized by spacing_norm (D, H, W).
    """
    # Get coordinates of non-zero pixels
    # mask shape: (D, H, W)
    pts_pred = np.argwhere(mask_pred)
    pts_gt = np.argwhere(mask_gt)

    # Handle empty cases
    # If both empty, distance is 0 (perfect match)
    if len(pts_pred) == 0 and len(pts_gt) == 0:
        return 0.0
    # If one is empty, assign max penalty (1.0 in normalized space)
    if len(pts_pred) == 0 or len(pts_gt) == 0:
        return 1.0

    # Normalize coordinates to [0, 1] unit cube
    # spacing_norm is (D, H, W)
    pts_pred = pts_pred.astype(float) / spacing_norm
    pts_gt = pts_gt.astype(float) / spacing_norm

    # Calculate directed Hausdorff distances
    # H(A, B) = max(h(A, B), h(B, A))
    d_pred_gt = directed_hausdorff(pts_pred, pts_gt)[0]
    d_gt_pred = directed_hausdorff(pts_gt, pts_pred)[0]

    return max(d_pred_gt, d_gt_pred)


def compute_dice_3d(mask_pred, mask_gt):
    """Computes 3D Dice Coefficient."""
    intersection = np.sum(mask_pred * mask_gt)
    sum_pred = np.sum(mask_pred)
    sum_gt = np.sum(mask_gt)

    if sum_pred + sum_gt == 0:
        return 1.0  # Both empty

    return (2.0 * intersection) / (sum_pred + sum_gt)


def validate_and_analyze():
    print("\n=== Starting Validation & Failure Analysis ===")
    device = Config.DEVICE
    set_seed(Config.SEED)

    # 1. Load Validation Dataset
    val_dataset = UWMadisonDataset(split="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Load Best Model
    model = DeepLabV3Plus(num_classes=Config.NUM_CLASSES).to(device)
    if os.path.exists(Config.CHECKPOINT_PATH):
        model.load_state_dict(torch.load(Config.CHECKPOINT_PATH, map_location=device))
        print(f"Loaded model from {Config.CHECKPOINT_PATH}")
    else:
        print("Error: Checkpoint not found! Using random weights for validation.")

    model.eval()

    # 3. Collect Predictions (Grouped by Case_Day)
    # structure: results[case_day] = {'slices': [], 'preds': [], 'gts': []}
    results = {}

    print("Collecting validation predictions...")
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            masks = batch["mask"].cpu().numpy()  # (B, 3, H, W)
            ids = batch["id"]

            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()  # (B, 3, H, W)

            # Threshold to binary
            preds_bin = (probs > Config.CONFIDENCE_THRESHOLD).astype(np.uint8)

            for i, id_str in enumerate(ids):
                # Parse ID: caseXXX_dayYY_slice_ZZZZ
                parts = id_str.split("_")
                case_day = f"{parts[0]}_{parts[1]}"
                # Slice ID is the 4th part (index 3)
                slice_num = int(parts[3])

                if case_day not in results:
                    results[case_day] = {"slices": [], "preds": [], "gts": []}

                results[case_day]["slices"].append(slice_num)
                results[case_day]["preds"].append(preds_bin[i])  # (3, H, W)
                results[case_day]["gts"].append(masks[i])  # (3, H, W)

    # 4. Compute 3D Metrics
    dice_scores = []
    hausdorff_scores = []
    combined_scores = []

    # For failure analysis
    case_metrics = []

    print("Computing 3D metrics and performing failure analysis...")
    for case_day, data in results.items():
        # Sort by slice index to form proper 3D volume
        sorted_indices = np.argsort(data["slices"])

        # Stack into (D, 3, H, W)
        vol_pred = np.stack(data["preds"])[sorted_indices]
        vol_gt = np.stack(data["gts"])[sorted_indices]

        # Transpose to (3, D, H, W) to iterate by class
        vol_pred = np.transpose(vol_pred, (1, 0, 2, 3))
        vol_gt = np.transpose(vol_gt, (1, 0, 2, 3))

        D, H, W = vol_pred.shape[1:]
        # Normalization factor for Hausdorff: [D, H, W]
        norm_factor = np.array([D, H, W])

        case_dices = []
        case_hds = []

        for c in range(3):  # 3 classes
            # Apply 3D cleanup (same as inference)
            cleaned_pred = remove_small_objects_3d(
                vol_pred[c], Config.MIN_VOLUME_THRESHOLD
            )

            d = compute_dice_3d(cleaned_pred, vol_gt[c])
            h = compute_hausdorff_3d(cleaned_pred, vol_gt[c], norm_factor)

            case_dices.append(d)
            case_hds.append(h)

        mean_dice = np.mean(case_dices)
        mean_hd = np.mean(case_hds)

        # Metric: 0.4*Dice + 0.6*(1 - HD)
        # We clamp HD to 1.0 to ensure the score doesn't go negative if distance > 1.0
        score = 0.4 * mean_dice + 0.6 * (1.0 - min(1.0, mean_hd))

        dice_scores.append(mean_dice)
        hausdorff_scores.append(mean_hd)
        combined_scores.append(score)

        case_metrics.append(
            {
                "case_day": case_day,
                "dice": mean_dice,
                "hd": mean_hd,
                "score": score,
                "depth": D,
            }
        )

    final_metric = np.mean(combined_scores)
    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_metric:.18f}")

    # 5. Failure Analysis
    df_analysis = pd.DataFrame(case_metrics)
    df_analysis["error"] = 1.0 - df_analysis["score"]

    print("\nFailure Analysis (Correlation with Error):")
    # Correlate error with depth (proxy for organ size/scan complexity)
    if len(df_analysis) > 1:
        corr_depth = df_analysis["error"].corr(df_analysis["depth"])
        print(f"Correlation between Error and Scan Depth: {corr_depth:.4f}")
    else:
        print("Not enough samples for correlation analysis.")

    # Identify worst cases
    print("\nWorst 5 Cases:")
    print(
        df_analysis.sort_values("score")
        .head(5)[["case_day", "score", "dice", "hd"]]
        .to_string(index=False)
    )


def main():
    # 1. Train
    # Using 5 epochs for fast baseline execution
    print("=== Starting Training Pipeline ===")
    run_training(epochs=5, batch_size=Config.BATCH_SIZE, load_cached_data=True)

    # 2. Validate & Analyze
    validate_and_analyze()

    # 3. Inference & Submission
    print("\n=== Starting Inference Pipeline ===")
    run_inference(load_cached_data=True)


if __name__ == "__main__":
    main()
