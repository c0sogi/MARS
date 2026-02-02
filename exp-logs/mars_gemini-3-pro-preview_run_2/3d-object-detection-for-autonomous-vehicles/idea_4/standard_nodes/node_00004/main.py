import os
import sys
import json
import math
import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

from library.config import Config
from library.train import train_model
from library.dataset import Mono3DDataset
from library.model import MonoCenterNet
from library.inference import generate_submission, decode_detections
import library.utils as utils


def parse_prediction_string(pred_str):
    """Parses the prediction string into a list of dictionaries."""
    if not pred_str or pd.isna(pred_str) or str(pred_str).strip() == "":
        return []

    parts = str(pred_str).strip().split()
    # Format: score x y z w l h yaw class
    num_fields = 9
    preds = []
    for i in range(0, len(parts), num_fields):
        try:
            if i + num_fields > len(parts):
                break
            p = {
                "score": float(parts[i]),
                "center_x": float(parts[i + 1]),
                "center_y": float(parts[i + 2]),
                "center_z": float(parts[i + 3]),
                "width": float(parts[i + 4]),
                "length": float(parts[i + 5]),
                "height": float(parts[i + 6]),
                "yaw": float(parts[i + 7]),
                "class_name": parts[i + 8],
            }
            preds.append(p)
        except ValueError:
            continue
    return preds


def calculate_3d_iou(box1, box2):
    """
    Calculates 3D IoU as defined in the task.
    IoU = (Intersection Ground * Intersection Height) / (Union Volume)
    """
    # 1. Ground Intersection (Rotated Rectangles)
    # OpenCV RotatedRect: ((cx, cy), (w, h), angle_deg)
    # Note: We use width as w, length as h. Angle in degrees.
    rect1 = (
        (box1["center_x"], box1["center_y"]),
        (box1["width"], box1["length"]),
        np.degrees(box1["yaw"]),
    )
    rect2 = (
        (box2["center_x"], box2["center_y"]),
        (box2["width"], box2["length"]),
        np.degrees(box2["yaw"]),
    )

    try:
        res, points = cv2.rotatedRectangleIntersection(rect1, rect2)
        if res == cv2.INTERSECT_NONE or points is None:
            inter_area = 0.0
        else:
            # points shape is (N, 1, 2)
            inter_area = cv2.contourArea(points.astype(np.float32))
    except Exception:
        inter_area = 0.0

    # 2. Height Intersection
    z1_min = box1["center_z"] - box1["height"] / 2.0
    z1_max = box1["center_z"] + box1["height"] / 2.0
    z2_min = box2["center_z"] - box2["height"] / 2.0
    z2_max = box2["center_z"] + box2["height"] / 2.0

    inter_h_min = max(z1_min, z2_min)
    inter_h_max = min(z1_max, z2_max)
    inter_h = max(0.0, inter_h_max - inter_h_min)

    # 3. Volumes
    inter_vol = inter_area * inter_h

    vol1 = box1["width"] * box1["length"] * box1["height"]
    vol2 = box2["width"] * box2["length"] * box2["height"]

    union_vol = vol1 + vol2 - inter_vol

    if union_vol <= 1e-6:
        return 0.0

    return inter_vol / union_vol


def evaluate_metric(val_loader, model, device):
    """
    Computes the mAP metric over the validation set.
    """
    model.eval()

    # Thresholds: 0.5 to 0.95 step 0.05
    thresholds = [round(x, 2) for x in np.arange(0.5, 0.96, 0.05)]

    image_precisions = []
    fa_data = []  # For failure analysis

    # Load GT metadata map for fast access
    print("Loading validation metadata for evaluation...")
    df_val = pd.read_csv(Config.VAL_METADATA)
    df_val["annotations"] = df_val["annotations"].apply(json.loads)
    gt_map = df_val.set_index("token")["annotations"].to_dict()

    print("Running Validation Inference...")

    with torch.no_grad():
        for batch_idx, (img, _, info) in enumerate(val_loader):
            img = img.to(device)
            outputs = model(img)

            # Decode predictions (Batch size 1)
            # Use a low threshold to capture potential matches
            preds_list = decode_detections(
                outputs, info, K=Config.TOP_K, conf_thresh=0.01
            )
            pred_str = preds_list[0]
            preds = parse_prediction_string(pred_str)

            # Sort preds by confidence descending (required by metric)
            preds.sort(key=lambda x: x["score"], reverse=True)

            # Get GT
            token = info["token"][0]
            gts = gt_map.get(token, [])

            # Calculate Average Precision for this image
            precisions = []

            for t in thresholds:
                tp = 0
                fp = 0

                matched_gt_indices = set()

                for p in preds:
                    best_iou = -1.0
                    best_gt_idx = -1

                    # Find best matching GT
                    for i, g in enumerate(gts):
                        if i in matched_gt_indices:
                            continue
                        if p["class_name"] != g["class_name"]:
                            continue

                        iou = calculate_3d_iou(p, g)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = i

                    if best_iou > t:
                        tp += 1
                        matched_gt_indices.add(best_gt_idx)

                        # Collect Failure Analysis Data (only at t=0.5 to avoid duplicates)
                        if t == 0.5:
                            g_match = gts[best_gt_idx]
                            dist_err = np.sqrt(
                                (p["center_x"] - g_match["center_x"]) ** 2
                                + (p["center_y"] - g_match["center_y"]) ** 2
                                + (p["center_z"] - g_match["center_z"]) ** 2
                            )
                            # Approx depth (distance from origin)
                            depth_val = np.sqrt(
                                g_match["center_x"] ** 2 + g_match["center_y"] ** 2
                            )
                            vol_val = (
                                g_match["width"] * g_match["length"] * g_match["height"]
                            )

                            fa_data.append(
                                {
                                    "dist_err": dist_err,
                                    "depth": depth_val,
                                    "volume": vol_val,
                                }
                            )
                    else:
                        fp += 1

                fn = len(gts) - len(matched_gt_indices)

                denom = tp + fp + fn
                if denom == 0:
                    prec = 1.0  # Perfect (No GT, No Preds)
                else:
                    prec = tp / denom

                precisions.append(prec)

            # Average precision over thresholds for this image
            avg_prec = sum(precisions) / len(precisions)
            image_precisions.append(avg_prec)

    final_metric = (
        sum(image_precisions) / len(image_precisions) if image_precisions else 0.0
    )
    print(f"Final Validation Metric: {final_metric:.18f}")

    return fa_data


def failure_analysis(fa_data):
    if not fa_data:
        print("No matched predictions found for failure analysis.")
        return

    df = pd.DataFrame(fa_data)
    print("\nFailure Analysis Correlations (Error vs Attribute):")

    if len(df) > 5:
        # Correlation: Distance Error vs Depth
        corr_depth, _ = pearsonr(df["dist_err"], df["depth"])
        print(f"Correlation (Distance Error vs Depth): {corr_depth:.4f}")

        # Correlation: Distance Error vs Volume
        corr_vol, _ = pearsonr(df["dist_err"], df["volume"])
        print(f"Correlation (Distance Error vs Volume): {corr_vol:.4f}")
    else:
        print("Insufficient data points for meaningful correlation analysis.")


def main():
    # 1. Setup
    Config.setup()
    Config.set_seed(42)
    device = Config.DEVICE

    # 2. Train
    # Using 5 epochs to ensure completion within time limit while getting reasonable results.
    print("=== Starting Training ===")
    train_model(
        debug=False, load_cached_data=True, num_epochs=5, batch_size=Config.BATCH_SIZE
    )

    # 3. Validation & Metric
    print("\n=== Starting Validation ===")

    # Load Model
    model = MonoCenterNet().to(device)
    ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(ckpt_path):
        print("Best model not found, using latest.")
        ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "latest_model.pth")

    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("No checkpoint found! Skipping evaluation.")
        return

    # Prepare Validation Loader
    val_dataset = Mono3DDataset(split="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=Config.NUM_WORKERS
    )

    # Evaluate
    fa_data = evaluate_metric(val_loader, model, device)

    # 4. Failure Analysis
    print("\n=== Starting Failure Analysis ===")
    failure_analysis(fa_data)

    # 5. Submission
    print("\n=== Generating Submission ===")
    generate_submission(ckpt_path, split="test", load_cached_data=True)


if __name__ == "__main__":
    main()
