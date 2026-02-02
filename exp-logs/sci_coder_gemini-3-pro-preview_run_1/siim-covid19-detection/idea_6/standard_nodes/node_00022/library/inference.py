import os
import cv2
import torch
import pydicom
import pandas as pd
import numpy as np
from library.config import Config
from library.model import ResNet34FPN
from library.data import prepare_test_loader


def get_original_dimensions(file_path):
    """
    Reads DICOM header to get original image dimensions.
    Used to scale bounding boxes back to original resolution.
    """
    full_path = os.path.join(Config.INPUT_DIR, file_path)
    try:
        dcm = pydicom.dcmread(full_path, stop_before_pixels=True)
        return dcm.Columns, dcm.Rows  # Width, Height
    except Exception as e:
        # Fallback to model input size if read fails (unlikely)
        return Config.IMG_SIZE, Config.IMG_SIZE


def mask_to_prediction_string(mask, orig_w, orig_h, threshold):
    """
    Converts a probability mask to the submission format string.
    Scales coordinates from Config.IMG_SIZE to original dimensions.
    """
    # Threshold mask to binary
    mask_binary = (mask > threshold).astype(np.uint8)

    if mask_binary.sum() == 0:
        return Config.NONE_PREDICTION

    # Find contours
    contours, _ = cv2.findContours(
        mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    predictions = []

    # Scaling factors (Original / Model Input)
    scale_w = orig_w / Config.IMG_SIZE
    scale_h = orig_h / Config.IMG_SIZE

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Filter very small artifacts in the resized space
        if w * h < 10:
            continue

        # Compute confidence as mean probability within the box
        box_mask = mask[y : y + h, x : x + w]
        conf = np.mean(box_mask) if box_mask.size > 0 else 0.0

        # Scale coordinates to original image space
        x1 = x * scale_w
        y1 = y * scale_h
        x2 = (x + w) * scale_w
        y2 = (y + h) * scale_h

        # Format: opacity conf xmin ymin xmax ymax
        predictions.append(f"opacity {conf:.4f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}")

    if not predictions:
        return Config.NONE_PREDICTION

    return " ".join(predictions)


def generate_submission(device=Config.DEVICE):
    """
    Generates the submission.csv file for the test set.
    """
    print("Generating submission...")

    # 1. Load Test Metadata
    if not os.path.exists(Config.TEST_CSV):
        print(f"Error: Test metadata not found at {Config.TEST_CSV}")
        return

    test_df = pd.read_csv(Config.TEST_CSV)
    print(f"Loaded test metadata: {len(test_df)} images")

    # 2. Prepare Data Loader
    # Note: prepare_test_loader handles caching internally
    test_loader = prepare_test_loader(test_df)

    # 3. Load Model
    model = ResNet34FPN()
    model.to(device)

    checkpoint_path = Config.CHECKPOINT_PATH
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(
            f"WARNING: Checkpoint not found at {checkpoint_path}. Using random weights (for debugging only)."
        )

    model.eval()

    # 4. Inference Loop
    results = []

    # Mapping for study labels (index -> name)
    study_class_map = {
        0: "negative",
        1: "typical",
        2: "indeterminate",
        3: "atypical",
    }

    with torch.no_grad():
        for batch_idx, (images, _, _) in enumerate(test_loader):
            images = images.to(device)

            # Forward Pass
            cls_logits, seg_logits = model(images)

            # Calculate Probabilities
            cls_probs = torch.softmax(cls_logits, dim=1)
            seg_probs = torch.sigmoid(seg_logits)

            # Convert to Numpy
            cls_preds_idx = torch.argmax(cls_probs, dim=1).cpu().numpy()
            cls_confs = torch.max(cls_probs, dim=1).values.cpu().numpy()
            seg_masks = seg_probs.cpu().numpy()  # (B, 1, H, W)

            batch_size = images.size(0)

            for i in range(batch_size):
                # Map back to DataFrame row using global index
                global_idx = batch_idx * Config.BATCH_SIZE + i
                if global_idx >= len(test_df):
                    break

                row = test_df.iloc[global_idx]
                study_id = row["study_id"]
                image_id = row["image_id"]
                file_path = row["file_path"]

                # --- Study Level Prediction ---
                pred_idx = cls_preds_idx[i]
                pred_label = study_class_map[pred_idx]
                pred_conf = cls_confs[i]

                # Format: class conf 0 0 1 1
                study_str = f"{pred_label} {pred_conf:.6f} 0 0 1 1"
                results.append(
                    {"Id": f"{study_id}_study", "PredictionString": study_str}
                )

                # --- Image Level Prediction ---
                # Gating Logic: If Study is Negative, force Image to 'none'
                if pred_label == "negative":
                    image_str = Config.NONE_PREDICTION
                else:
                    # Get Original Dimensions for Scaling
                    orig_w, orig_h = get_original_dimensions(file_path)

                    # Process Mask
                    mask = seg_masks[i, 0]
                    image_str = mask_to_prediction_string(
                        mask, orig_w, orig_h, threshold=Config.SEG_THRESHOLD
                    )

                results.append(
                    {"Id": f"{image_id}_image", "PredictionString": image_str}
                )

    # 5. Save Submission
    submission_df = pd.DataFrame(results)

    # Ensure directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Total predictions: {len(submission_df)}")
