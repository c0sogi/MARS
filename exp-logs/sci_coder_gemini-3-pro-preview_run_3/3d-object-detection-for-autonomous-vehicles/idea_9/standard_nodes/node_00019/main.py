import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import provided library modules
from library.config import Config
from library.dataset import LidarDataset, collate_fn
from library.detector import PointPillarsDetector
from library.utils import box_decode, nms_3d, iou3d


def main():
    # ==============================================================================
    # 1. Configuration & Setup
    # ==============================================================================
    # Override Config for a fast baseline execution
    Config.EPOCHS = 3
    Config.SUBSET_SIZE = 1200  # Train on a subset to ensure < 2h runtime
    Config.BATCH_SIZE = 4

    # Ensure reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # ==============================================================================
    # 2. Data Loading
    # ==============================================================================
    # Train on subset
    train_ds = LidarDataset(split="train", subset_size=Config.SUBSET_SIZE)
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Validate on FULL set to get accurate metric
    val_ds = LidarDataset(split="val", subset_size=None)
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # ==============================================================================
    # 3. Training
    # ==============================================================================
    detector = PointPillarsDetector()

    # Execute training loop (saves 'model_checkpoint.pth' based on val loss)
    detector.train(train_loader, val_loader)

    # Reload the best model for final evaluation
    checkpoint_path = os.path.join(Config.WORKING_DIR, "model_checkpoint.pth")
    if os.path.exists(checkpoint_path):
        detector.load_checkpoint(checkpoint_path)
    else:
        print("Warning: Checkpoint not found, using current model state.")

    # ==============================================================================
    # 4. Evaluation (Metric Calculation)
    # ==============================================================================
    print("Starting evaluation on full validation set...")
    final_metric, analysis_data = evaluate_map(detector, val_loader, device)

    # Print the required metric string
    print(f"Final Validation Metric: {final_metric}")

    # ==============================================================================
    # 5. Failure Analysis
    # ==============================================================================
    perform_failure_analysis(analysis_data)

    # ==============================================================================
    # 6. Submission
    # ==============================================================================
    # Threshold from task description
    submission_threshold = 0.06434981603098806

    if final_metric > submission_threshold:
        print("Metric threshold met. Generating submission...")
        test_ds = LidarDataset(split="test", subset_size=None)
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        detector.generate_submission(test_loader, Config.SUBMISSION_PATH)
    else:
        print(
            f"Metric {final_metric} did not meet threshold {submission_threshold}. Skipping submission."
        )


def evaluate_map(detector, dataloader, device):
    """
    Calculates the Mean Average Precision at IoU thresholds 0.5:0.05:0.95.
    Also collects data for failure analysis.
    """
    detector.model.eval()
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    image_scores = []

    # Data for failure analysis
    analysis_data = {
        "error": [],  # 1.0 - AP
        "num_points": [],  # Input point count
        "num_gt": [],  # Number of GT objects
    }

    with torch.no_grad():
        # Disable tqdm for silent execution as requested
        for batch in tqdm(dataloader, disable=True):
            voxels = batch["voxels"].to(device)
            num_points = batch["num_points"].to(device)
            coordinates = batch["coordinates"].to(device)
            batch_size = len(batch["sample_tokens"])

            # --- Inference ---
            hm_pred, reg_pred, feats = detector.model(
                voxels, num_points, coordinates, batch_size=batch_size
            )

            # Stage 1 Proposals
            proposals, scores, cls_ids = detector.model.center_head.get_proposals(
                hm_pred, reg_pred, topk=200
            )

            # Stage 2 Refinement & Rectification
            residuals, iou_pred = detector.model.forward_stage2(feats, proposals)

            refined_boxes = box_decode(residuals, proposals)
            # Rectify scores: Score = Cls * IoU^alpha
            rectified_scores = scores * torch.pow(
                iou_pred.squeeze(-1), Config.IOU_RECT_ALPHA
            )

            # --- Metric Calculation per Sample ---
            for i in range(batch_size):
                # 1. Prepare Predictions
                boxes = refined_boxes[i]
                sc = rectified_scores[i]

                # Apply NMS
                keep = nms_3d(boxes, sc, iou_threshold=0.1)
                pred_boxes = boxes[keep]
                pred_scores = sc[keep]

                # 2. Prepare Ground Truth
                gt_boxes = batch["gt_boxes"][i].to(device)

                # 3. Calculate AP for this image
                precisions = []

                # Handle cases with no GT
                if len(gt_boxes) == 0:
                    analysis_data["num_gt"].append(0)
                    if len(pred_boxes) == 0:
                        # Perfect match (empty == empty)
                        precisions = [1.0] * len(thresholds)
                    else:
                        # False Positives exist, score is 0
                        precisions = [0.0] * len(thresholds)
                else:
                    analysis_data["num_gt"].append(len(gt_boxes))

                    # Pre-calculate IoU matrix if predictions exist
                    iou_mat = None
                    if len(pred_boxes) > 0:
                        iou_mat = iou3d(pred_boxes, gt_boxes)  # (N_pred, N_gt)

                    for t in thresholds:
                        if len(pred_boxes) == 0:
                            # FN for all GTs -> Precision = 0
                            precisions.append(0.0)
                            continue

                        # Match predictions to GT
                        # Sort predictions by score (already sorted by NMS usually, but ensure)
                        sort_idx = torch.argsort(pred_scores, descending=True)
                        sorted_iou_mat = iou_mat[sort_idx]

                        tp = 0
                        fp = 0
                        matched_gt_mask = torch.zeros(
                            len(gt_boxes), dtype=torch.bool, device=device
                        )

                        # Greedy matching
                        for p_idx in range(len(pred_boxes)):
                            # Get IoUs for this prediction
                            ious = sorted_iou_mat[p_idx]

                            # Mask already matched GTs to find best unmatched
                            # We use a temporary clone to mask out matched
                            valid_ious = ious.clone()
                            valid_ious[matched_gt_mask] = -1.0

                            max_iou, max_gt_idx = torch.max(valid_ious, dim=0)

                            if max_iou > t:
                                tp += 1
                                matched_gt_mask[max_gt_idx] = True
                            else:
                                fp += 1

                        fn = len(gt_boxes) - matched_gt_mask.sum().item()
                        denom = tp + fp + fn

                        if denom == 0:
                            precisions.append(1.0)
                        else:
                            precisions.append(tp / denom)

                # Average Precision for this image
                ap_img = np.mean(precisions)
                image_scores.append(ap_img)

                # --- Collect Failure Analysis Data ---
                analysis_data["error"].append(1.0 - ap_img)

                # Estimate number of points for this sample
                # coordinates is (batch_idx, z, y, x), num_points is (M,)
                batch_mask = coordinates[:, 0] == i
                total_points = num_points[batch_mask].sum().item()
                analysis_data["num_points"].append(total_points)

    return np.mean(image_scores), analysis_data


def perform_failure_analysis(data):
    """
    Correlates error magnitude with input features.
    """
    df = pd.DataFrame(data)

    # Calculate correlations
    if len(df) > 1 and df["num_points"].std() > 0:
        corr_points = df["error"].corr(df["num_points"])
    else:
        corr_points = 0.0

    if len(df) > 1 and df["num_gt"].std() > 0:
        corr_gt = df["error"].corr(df["num_gt"])
    else:
        corr_gt = 0.0

    print("-" * 30)
    print("Failure Analysis (Correlation with Error Magnitude):")
    print(f"Input Feature: Num_Points     | Correlation: {corr_points:.4f}")
    print(f"Input Feature: Num_GT_Objects | Correlation: {corr_gt:.4f}")
    print("-" * 30)


if __name__ == "__main__":
    main()
