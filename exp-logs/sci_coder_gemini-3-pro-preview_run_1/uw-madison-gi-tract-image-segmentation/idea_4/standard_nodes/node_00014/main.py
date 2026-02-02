import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import binary_erosion

# Add current directory to path
sys.path.append(".")

# Import from library
from library.config import Config
from library.train import Trainer
from library.inference import InferencePipeline
from library.utils import (
    set_seed,
    load_and_preprocess_metadata,
    group_metadata_by_case,
    rle_decode,
    rle_encode,
)
from library.dataset import UWDataset, get_transforms
from library.model import UnetPlusPlus

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def get_surface(mask):
    """
    Extracts the surface pixels of a 3D binary mask to speed up Hausdorff calculation.
    """
    if np.sum(mask) == 0:
        return mask
    # Surface = Mask - Erode(Mask)
    eroded = binary_erosion(mask, iterations=1)
    surface = mask ^ eroded
    return surface


def compute_metrics_3d(pred_vol, gt_vol, shape):
    """
    Computes Dice and Normalized 3D Hausdorff Score for a single class.

    Args:
        pred_vol (np.ndarray): Predicted binary volume (D, H, W).
        gt_vol (np.ndarray): Ground truth binary volume (D, H, W).
        shape (tuple): (D, H, W) for normalization.

    Returns:
        tuple: (dice_score, hausdorff_score)
    """
    # 1. Dice Coefficient
    # Prompt: "Dice defined to be 0 when both X and Y are empty"
    inter = np.sum(pred_vol * gt_vol)
    union = np.sum(pred_vol) + np.sum(gt_vol)

    if union == 0:
        dice = 0.0
    else:
        dice = (2.0 * inter) / union

    # 2. Hausdorff Distance
    # Prompt: "normalized by image size to create a bounded 0-1 score"
    # We interpret this as: Score = 1 - HD(normalized_coords)

    if np.sum(pred_vol) == 0 and np.sum(gt_vol) == 0:
        # If both empty, distance is 0, score is 1.
        hd_score = 1.0
    elif np.sum(pred_vol) == 0 or np.sum(gt_vol) == 0:
        # If one empty, max penalty.
        hd_score = 0.0
    else:
        # Optimization: Use surface points only
        pred_surf = get_surface(pred_vol)
        gt_surf = get_surface(gt_vol)

        # Get coordinates (z, y, x)
        pred_coords = np.argwhere(pred_surf).astype(np.float32)
        gt_coords = np.argwhere(gt_surf).astype(np.float32)

        if len(pred_coords) == 0:
            pred_coords = np.argwhere(pred_vol).astype(np.float32)
        if len(gt_coords) == 0:
            gt_coords = np.argwhere(gt_vol).astype(np.float32)

        # Normalize coordinates to unit cube [0, 1]
        # shape is (D, H, W)
        scale = np.array(shape, dtype=np.float32)
        pred_coords /= scale
        gt_coords /= scale

        # Compute Directed Hausdorff
        d1 = directed_hausdorff(pred_coords, gt_coords)[0]
        d2 = directed_hausdorff(gt_coords, pred_coords)[0]
        hd_dist = max(d1, d2)

        # Bounded 0-1 score
        hd_score = max(0.0, 1.0 - hd_dist)

    return dice, hd_score


def validate_and_analyze(model, val_df, device):
    """
    Performs validation on the hold-out set, computes the official metric,
    and runs failure analysis.
    """
    print("Starting Validation and Failure Analysis...")
    model.eval()
    transforms = get_transforms(mode="val")

    # Group by case for 3D processing
    grouped_cases = group_metadata_by_case(val_df)

    metrics = []
    case_stats = []

    # Inference Pipeline for post-processing logic
    pipeline = InferencePipeline()

    with torch.no_grad():
        for case_key, case_df in grouped_cases.items():
            case_id = f"{case_key[0]}_{case_key[1]}"

            # Setup loader
            ds = UWDataset(case_df, mode="val", transforms=transforms)
            loader = DataLoader(
                ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4
            )

            # 1. Predict Volume
            preds_list = []
            gt_list = []

            for batch in loader:
                imgs = batch["image"].to(device)
                masks = batch["mask"].cpu().numpy()  # (B, C, H, W)

                logits = model(imgs)
                probs = torch.sigmoid(logits)

                # Resize to original resolution (if needed, but val is usually resized in transforms)
                # Config.IMG_SIZE is used. We assume validation GT is also resized or we resize preds.
                # UWDataset returns masks resized to IMG_SIZE. We will evaluate at IMG_SIZE.

                preds_list.append(probs.cpu())
                gt_list.append(masks)

            # Concatenate to (D, C, H, W)
            vol_probs = torch.cat(preds_list, dim=0)
            vol_gt = np.concatenate(gt_list, axis=0)

            # Threshold
            vol_mask = (vol_probs > 0.5).numpy().astype(np.uint8)

            D, C, H, W = vol_mask.shape
            shape = (D, H, W)

            case_scores = []

            # 2. Process per class
            for c_idx, cls_name in enumerate(Config.CLASSES):
                pred_c = vol_mask[:, c_idx, :, :]
                gt_c = vol_gt[:, c_idx, :, :]

                # Post-process (CCA) - Critical for Hausdorff
                pred_c_clean = pipeline.post_process_volume(pred_c)

                # Compute Metrics
                dice, hd = compute_metrics_3d(pred_c_clean, gt_c, shape)

                # Combined Metric: 0.4*Dice + 0.6*HD
                score = 0.4 * dice + 0.6 * hd
                case_scores.append(score)

                # Store for analysis
                organ_size = np.sum(gt_c)
                metrics.append(
                    {
                        "case": case_id,
                        "class": cls_name,
                        "dice": dice,
                        "hd_score": hd,
                        "final_score": score,
                        "organ_size": organ_size,
                        "depth": D,
                    }
                )

            case_stats.append(np.mean(case_scores))

    # Aggregate
    df_metrics = pd.DataFrame(metrics)
    final_metric = df_metrics["final_score"].mean()

    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\n=== Failure Analysis ===")
    # Correlation between Error (1-Score) and Organ Size
    df_metrics["error"] = 1.0 - df_metrics["final_score"]

    # Handle NaN correlations if constant
    if df_metrics["organ_size"].std() > 0:
        corr_size = df_metrics["error"].corr(df_metrics["organ_size"])
        print(f"Correlation (Error vs Organ Size): {corr_size:.4f}")
        if corr_size < -0.3:
            print("  -> Observation: Model performs better on larger organs.")
        elif corr_size > 0.3:
            print("  -> Observation: Model performs worse on larger organs.")

    # Performance by Class
    print("\nPerformance by Class:")
    print(df_metrics.groupby("class")[["dice", "hd_score", "final_score"]].mean())

    return final_metric


def main():
    # 1. Configuration Override for Fast Baseline
    set_seed(Config.SEED)
    Config.EPOCHS = 10  # Reduced for speed (Baseline)
    Config.BATCH_SIZE = 48  # Increased for A100 efficiency

    print(f"Running experiment: {Config.EXP_NAME}")
    print(f"Device: {Config.DEVICE}")

    # 2. Training
    trainer = Trainer()
    # Fit the model (internally handles data loading and training loop)
    best_dice_val = trainer.fit()
    print(f"Training finished. Best Slice-Dice: {best_dice_val:.4f}")

    # 3. Validation (Official Metric)
    # Load validation metadata
    if not os.path.exists(Config.VAL_CSV):
        print("Validation metadata not found.")
        return

    val_df = load_and_preprocess_metadata(Config.VAL_CSV)

    # Load best model
    model = UnetPlusPlus(
        classes=Config.NUM_CLASSES, deep_supervision=Config.DEEP_SUPERVISION
    )
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path, map_location=Config.DEVICE, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best model from epoch {checkpoint['epoch']}")
    else:
        print("Checkpoint not found, using initialized model (random weights).")

    model.to(Config.DEVICE)

    # Run Validation
    final_metric = validate_and_analyze(model, val_df, Config.DEVICE)

    # 4. Submission
    THRESHOLD = 0.5184837797359911
    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric:.6f}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        pipeline = InferencePipeline(model_path=checkpoint_path)
        pipeline.generate_submission()

        # Verify file location
        expected_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        if os.path.exists(expected_path):
            print(f"Submission verified at: {expected_path}")
        else:
            print("Error: Submission file not found after generation.")
    else:
        print(
            f"\nMetric ({final_metric:.6f}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
