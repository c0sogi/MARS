import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
from scipy.stats import pearsonr
from ultralytics import YOLO
from collections import defaultdict

# Import library functions
from library.config import (
    seed_everything,
    SEED,
    VAL_METADATA_PATH,
    INPUT_DIR,
    IMG_SIZE,
    CLASS_ID_TO_NAME,
    SUBMISSION_DIR,
)
from library.train_engine import train_model
from library.inference import generate_submission
from library.dicom_utils import process_dicom_image

# =============================================================================
# CONFIGURATION
# =============================================================================
# Fast baseline settings
TRAIN_SAMPLE_SIZE = None  # Use full dataset (Cite Lesson 00008)
TRAIN_EPOCHS = 15  # Adjusted for full dataset
IOU_THRESHOLD_METRIC = 0.4
CONF_THRESHOLD_VAL = 0.01
BASELINE_MAP = 0.2646659736645272

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0
    return intersection / union


def compute_ap(recalls, precisions):
    """
    Compute Average Precision using the PASCAL VOC method (all-point interpolation).
    """
    # Append sentinel values at the end
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Integrate area under curve
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def validate_and_analyze(model_path, val_csv_path):
    """
    Runs validation, computes mAP@0.4, and performs failure analysis.
    """
    print("\n==== Starting Validation & Failure Analysis ====")

    # Load Metadata
    df_val = pd.read_csv(val_csv_path)
    unique_images = df_val["image_id"].unique()

    # Load Model
    model = YOLO(model_path)

    # Structures for Metric Calculation
    # class_id -> list of {'conf': float, 'iou_max': float, 'tp': bool}
    predictions_per_class = defaultdict(list)
    # class_id -> total ground truth count
    gt_counts = defaultdict(int)

    # Structures for Failure Analysis
    # list of {'error': float, 'area': float, 'aspect_ratio': float}
    failure_data = []

    print(f"Validating on {len(unique_images)} images...")

    # Inference Loop
    # We process image by image
    for img_id in unique_images:
        # Get GT for this image
        img_gt = df_val[df_val["image_id"] == img_id]

        # Load Image
        rel_path = img_gt.iloc[0]["file_path"]
        dicom_path = os.path.join(INPUT_DIR, rel_path)

        # Read original dims for scaling GT
        try:
            ds = import_pydicom_safe(dicom_path)
            orig_h, orig_w = ds.Rows, ds.Columns
        except:
            # Fallback
            orig_h, orig_w = IMG_SIZE, IMG_SIZE

        # Preprocess
        img_array = process_dicom_image(dicom_path, target_size=IMG_SIZE)
        if len(img_array.shape) == 2:
            img_rgb = np.stack([img_array] * 3, axis=-1)
        else:
            img_rgb = img_array

        # Predict
        results = model.predict(
            img_rgb, conf=CONF_THRESHOLD_VAL, verbose=False, imgsz=IMG_SIZE
        )[0]

        # Parse Predictions
        preds = []
        if len(results.boxes) > 0:
            for box in results.boxes:
                # Scale boxes back to original dimensions
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().item()
                cls = int(box.cls[0].cpu().item())

                scale_x = orig_w / IMG_SIZE
                scale_y = orig_h / IMG_SIZE

                x1_o = x1 * scale_x
                y1_o = y1 * scale_y
                x2_o = x2 * scale_x
                y2_o = y2 * scale_y

                preds.append(
                    {"class_id": cls, "conf": conf, "box": [x1_o, y1_o, x2_o, y2_o]}
                )
        else:
            # Model predicts "No finding"
            # Represent as class 14 with 1-pixel box
            preds.append({"class_id": 14, "conf": 1.0, "box": [0, 0, 1, 1]})

        # Parse Ground Truth
        gts = []
        for _, row in img_gt.iterrows():
            cid = int(row["class_id"])
            gt_counts[cid] += 1

            if cid == 14:
                # No finding GT
                box = [0, 0, 1, 1]
            else:
                box = [row["x_min"], row["y_min"], row["x_max"], row["y_max"]]

            gts.append({"class_id": cid, "box": box, "matched": False})

            # For failure analysis (only for actual findings, not class 14)
            if cid != 14:
                w = box[2] - box[0]
                h = box[3] - box[1]
                area = w * h
                ar = w / h if h > 0 else 0

                # Find best matching prediction for this specific GT object
                best_iou_for_obj = 0.0
                for p in preds:
                    if p["class_id"] == cid:
                        iou = calculate_iou(box, p["box"])
                        best_iou_for_obj = max(best_iou_for_obj, iou)

                # Error Magnitude = 1 - IoU
                error = 1.0 - best_iou_for_obj
                failure_data.append({"error": error, "area": area, "aspect_ratio": ar})

        # Match Predictions to GT for mAP
        # Sort preds by confidence
        preds.sort(key=lambda x: x["conf"], reverse=True)

        for p in preds:
            p_cls = p["class_id"]
            p_box = p["box"]

            best_iou = 0.0
            best_gt_idx = -1

            # Find best matching GT of same class
            for idx, gt in enumerate(gts):
                if gt["class_id"] == p_cls:
                    iou = calculate_iou(p_box, gt["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

            # Determine TP/FP
            is_tp = False
            if best_iou > IOU_THRESHOLD_METRIC:
                if not gts[best_gt_idx]["matched"]:
                    gts[best_gt_idx]["matched"] = True
                    is_tp = True
                else:
                    # Duplicate detection
                    is_tp = False
            else:
                # Low IoU or no GT
                is_tp = False

            predictions_per_class[p_cls].append(
                {"conf": p["conf"], "tp": 1 if is_tp else 0}
            )

    # --- Calculate mAP ---
    aps = []
    # Iterate over all possible classes (0-14)
    for cid in range(15):
        if cid not in gt_counts or gt_counts[cid] == 0:
            continue

        class_preds = predictions_per_class[cid]
        # Sort by confidence descending
        class_preds.sort(key=lambda x: x["conf"], reverse=True)

        tps = np.array([x["tp"] for x in class_preds])
        fps = 1 - tps

        tp_cumsum = np.cumsum(tps)
        fp_cumsum = np.cumsum(fps)

        n_gt = gt_counts[cid]

        recalls = tp_cumsum / n_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

        ap = compute_ap(recalls, precisions)
        aps.append(ap)

    mAP = np.mean(aps) if aps else 0.0
    print(f"Final Validation Metric: {mAP}")

    # --- Failure Analysis ---
    print("\n==== Failure Analysis ====")
    if failure_data:
        df_fail = pd.DataFrame(failure_data)

        # Correlation: Error vs Area
        corr_area, _ = pearsonr(df_fail["error"], df_fail["area"])
        print(f"Correlation (Error Magnitude vs BBox Area): {corr_area:.4f}")

        # Correlation: Error vs Aspect Ratio
        corr_ar, _ = pearsonr(df_fail["error"], df_fail["aspect_ratio"])
        print(f"Correlation (Error Magnitude vs Aspect Ratio): {corr_ar:.4f}")

        # Interpretation
        if corr_area < -0.1:
            print("Observation: Model performs better on larger objects.")
        elif corr_area > 0.1:
            print("Observation: Model performs better on smaller objects.")
        else:
            print("Observation: No strong correlation with object size.")
    else:
        print("No finding objects found in validation set for failure analysis.")

    return mAP


def import_pydicom_safe(path):
    import pydicom

    return pydicom.dcmread(path, stop_before_pixels=True)


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    seed_everything(SEED)

    print(
        f"==== Starting Pipeline (Sample Size: {TRAIN_SAMPLE_SIZE}, Epochs: {TRAIN_EPOCHS}) ===="
    )

    # 1. Train
    # Using the library function. It handles data prep internally.
    best_weights_path = train_model(
        epochs=TRAIN_EPOCHS, debug_sample_size=TRAIN_SAMPLE_SIZE, load_cached_data=True
    )

    # 2. Validate & Analyze
    # We perform this manually to get the specific metric and analysis required
    mAP = validate_and_analyze(best_weights_path, VAL_METADATA_PATH)

    # 3. Submission
    if mAP > BASELINE_MAP:
        print(f"\n==== Generating Submission (mAP {mAP:.5f} > {BASELINE_MAP:.5f}) ====")
        generate_submission(best_weights_path)
    else:
        print(f"\n==== Skipping Submission (mAP {mAP:.5f} <= {BASELINE_MAP:.5f}) ====")

    print("\nPipeline Complete.")


if __name__ == "__main__":
    main()
