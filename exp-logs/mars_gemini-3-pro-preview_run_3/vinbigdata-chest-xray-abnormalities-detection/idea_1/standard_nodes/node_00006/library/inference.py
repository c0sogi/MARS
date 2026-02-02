import os
import pandas as pd
import numpy as np
import pydicom
import torch
from ultralytics import YOLO

from library.config import (
    INPUT_DIR,
    TEST_METADATA_PATH,
    SUBMISSION_PATH,
    IMG_SIZE,
    CONF_THRESHOLD,
    IOU_THRESHOLD,
    SEED,
    seed_everything,
)
from library.dicom_utils import process_dicom_image


def generate_submission(
    weights_path, test_metadata_path=TEST_METADATA_PATH, output_path=SUBMISSION_PATH
):
    """
    Generates predictions for the test set using the trained YOLO model.

    Args:
        weights_path (str): Path to the trained model weights (.pt file).
        test_metadata_path (str): Path to the test metadata CSV.
        output_path (str): Path to save the submission CSV.
    """
    # 1. Set Seed for Reproducibility
    seed_everything(SEED)

    # 2. Validate Paths
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found at {weights_path}")

    if not os.path.exists(test_metadata_path):
        raise FileNotFoundError(f"Test metadata not found at {test_metadata_path}")

    # 3. Load Model
    print(f"Loading YOLO model from {weights_path}...")
    model = YOLO(weights_path)

    # 4. Load Test Metadata
    df_test = pd.read_csv(test_metadata_path)
    print(f"Loaded {len(df_test)} test images from metadata.")

    results_list = []

    # 5. Run Inference
    print("Starting inference on test set...")

    # Disable gradient calculation for inference efficiency
    with torch.no_grad():
        for _, row in df_test.iterrows():
            image_id = row["image_id"]
            rel_path = row["file_path"]
            dicom_path = os.path.join(INPUT_DIR, rel_path)

            # --- Preprocessing ---
            # We need original dimensions to scale boxes back to the original image size.
            # Default to IMG_SIZE to avoid division by zero in case of read failure.
            orig_h, orig_w = IMG_SIZE, IMG_SIZE

            try:
                # Read only the header first to get dimensions
                ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
                orig_h, orig_w = ds.Rows, ds.Columns
            except Exception:
                # If header read fails, process_dicom_image handles the file read safely
                # and returns a black image, likely resulting in no detections.
                pass

            # Get resized and normalized image (IMG_SIZE x IMG_SIZE)
            img_array = process_dicom_image(dicom_path, target_size=IMG_SIZE)

            # Convert to RGB (H, W, 3) as YOLO expects 3 channels
            if len(img_array.shape) == 2:
                img_rgb = np.stack([img_array] * 3, axis=-1)
            else:
                img_rgb = img_array

            # --- Prediction ---
            # conf: Confidence threshold to filter weak detections
            # iou: IoU threshold for Non-Maximum Suppression (NMS)
            # verbose=False: Suppress per-image logging
            results = model.predict(
                img_rgb,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                verbose=False,
                imgsz=IMG_SIZE,
            )

            # --- Post-processing ---
            # results is a list (one per image in batch), we process one image at a time here.
            result = results[0]

            # Check if any detections exist
            if len(result.boxes) == 0:
                # Class 14: No finding (Format: 14 1 0 0 1 1)
                prediction_string = "14 1 0 0 1 1"
            else:
                # Iterate through detections
                temp_preds = []

                # Access boxes, confidence, and classes
                boxes = result.boxes
                for i in range(len(boxes)):
                    box = boxes[i]
                    # Get coordinates in [x1, y1, x2, y2] format
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().item()
                    cls_id = int(box.cls[0].cpu().item())

                    # --- Coordinate Rescaling ---
                    # The image was resized from (orig_w, orig_h) to (IMG_SIZE, IMG_SIZE)
                    # Note: process_dicom_image uses cv2.resize which stretches the image.
                    # We must scale X and Y independently.
                    scale_x = orig_w / IMG_SIZE
                    scale_y = orig_h / IMG_SIZE

                    x1_orig = x1 * scale_x
                    y1_orig = y1 * scale_y
                    x2_orig = x2 * scale_x
                    y2_orig = y2 * scale_y

                    # Clip coordinates to image boundaries to be safe
                    x1_orig = max(0, min(orig_w, x1_orig))
                    y1_orig = max(0, min(orig_h, y1_orig))
                    x2_orig = max(0, min(orig_w, x2_orig))
                    y2_orig = max(0, min(orig_h, y2_orig))

                    # Format: class_id confidence xmin ymin xmax ymax
                    pred_str = f"{cls_id} {conf:.4f} {x1_orig:.1f} {y1_orig:.1f} {x2_orig:.1f} {y2_orig:.1f}"
                    temp_preds.append(pred_str)

                # Join all detections for this image with a space
                prediction_string = " ".join(temp_preds)

            results_list.append(
                {"image_id": image_id, "PredictionString": prediction_string}
            )

    # 6. Save Submission
    df_submission = pd.DataFrame(results_list)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
