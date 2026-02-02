import os
import random
import numpy as np
import torch
import cv2
import pandas as pd
from torchvision.ops import box_iou
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mask2bbox(mask, threshold=0.5):
    """
    Converts a segmentation mask to a list of bounding boxes.

    Args:
        mask (np.ndarray or torch.Tensor): The prediction mask (H, W).
        threshold (float): Threshold for binarization.

    Returns:
        list: A list of bounding boxes in [xmin, ymin, xmax, ymax] format.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    # Binarize the mask
    binary_mask = (mask > threshold).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for contour in contours:
        # cv2.boundingRect returns (x, y, w, h)
        x, y, w, h = cv2.boundingRect(contour)

        # Convert to (xmin, ymin, xmax, ymax)
        xmin = x
        ymin = y
        xmax = x + w
        ymax = y + h

        # Filter out extremely small artifacts if necessary,
        # but generally we accept all contours that form a box.
        if w > 0 and h > 0:
            boxes.append([xmin, ymin, xmax, ymax])

    return boxes


def get_map_score(pred_boxes, pred_scores, true_boxes, iou_threshold=0.5):
    """
    Calculates the Average Precision (AP) for a single class (e.g., Opacity)
    using the PASCAL VOC 2010 method (all-point interpolation).

    Args:
        pred_boxes (list): List of lists of predicted boxes [xmin, ymin, xmax, ymax] for each image.
        pred_scores (list): List of lists of confidence scores for each predicted box.
        true_boxes (list): List of lists of ground truth boxes [xmin, ymin, xmax, ymax] for each image.
        iou_threshold (float): IoU threshold for matching predictions to ground truth.

    Returns:
        float: The Average Precision (AP) score.
    """
    if len(pred_boxes) != len(true_boxes):
        raise ValueError("Length of predictions and targets must match.")

    # Flatten all detections across images into a single list for sorting
    # Structure: {'score': float, 'image_idx': int, 'box': list}
    detections = []
    for i, (boxes, scores) in enumerate(zip(pred_boxes, pred_scores)):
        for box, score in zip(boxes, scores):
            detections.append({"score": score, "image_idx": i, "box": box})

    # Sort detections by confidence score (descending)
    detections.sort(key=lambda x: x["score"], reverse=True)

    # Initialize True Positive (TP) and False Positive (FP) arrays
    num_detections = len(detections)
    TP = np.zeros(num_detections)
    FP = np.zeros(num_detections)

    # Track matched ground truth boxes to ensure 1-to-1 matching
    # matched_gt[image_idx] = set(gt_box_indices_already_matched)
    matched_gt = {i: set() for i in range(len(true_boxes))}

    # Calculate total number of ground truth positives
    total_positives = sum(len(boxes) for boxes in true_boxes)

    if total_positives == 0:
        # If no ground truth objects exist, any detection is a False Positive.
        # AP is 0.0 if there are detections, or undefined (handled as 0.0) if no detections.
        return 0.0

    for i, detection in enumerate(detections):
        img_idx = detection["image_idx"]
        pred_box = torch.tensor([detection["box"]], dtype=torch.float32)

        gt_boxes = true_boxes[img_idx]

        if len(gt_boxes) == 0:
            FP[i] = 1
            continue

        gt_boxes_tensor = torch.tensor(gt_boxes, dtype=torch.float32)

        # Calculate IoU matrix between the predicted box and all GT boxes in the image
        iou_matrix = box_iou(pred_box, gt_boxes_tensor)

        # Find the GT box with the maximum IoU
        max_iou, max_idx = torch.max(iou_matrix, dim=1)
        max_iou = max_iou.item()
        max_idx = max_idx.item()

        if max_iou >= iou_threshold:
            if max_idx not in matched_gt[img_idx]:
                TP[i] = 1
                matched_gt[img_idx].add(max_idx)
            else:
                FP[i] = 1  # Duplicate detection (already matched this GT)
        else:
            FP[i] = 1  # IoU too low

    # Compute cumulative TP and FP
    acc_TP = np.cumsum(TP)
    acc_FP = np.cumsum(FP)

    # Compute Precision and Recall
    # Precision = TP / (TP + FP)
    # Recall = TP / Total Positives
    precisions = np.divide(
        acc_TP,
        (acc_TP + acc_FP),
        out=np.zeros_like(acc_TP),
        where=(acc_TP + acc_FP) != 0,
    )
    recalls = acc_TP / total_positives

    # Compute AP using VOC 2010 all-point interpolation
    # Append sentinel values to recall and precision arrays
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute the precision envelope (max precision to the right)
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Integrate area under the curve
    # Find indices where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # Sum of rectangular areas
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return float(ap)


def post_process_submission(
    study_preds, study_ids, image_preds, image_ids, save_path=Config.SUBMISSION_FILE
):
    """
    Formats predictions into the required submission string format and saves to CSV.

    Args:
        study_preds (list): List of arrays/lists containing probabilities for the 4 study classes.
                            Order must match Config.STUDY_LABELS:
                            [Negative, Typical, Indeterminate, Atypical].
        study_ids (list): List of study IDs (e.g., 'id_study').
        image_preds (list): List of dicts for each image. Each dict contains:
                            'boxes': list of [xmin, ymin, xmax, ymax]
                            'scores': list of floats
        image_ids (list): List of image IDs (e.g., 'id_image').
        save_path (str): Path to save the CSV file.

    Returns:
        pd.DataFrame: The submission dataframe.
    """
    rows = []

    # 1. Process Study Predictions
    # Map class indices to the required submission class strings
    # Config.STUDY_LABELS: ["Negative for Pneumonia", "Typical Appearance", "Indeterminate Appearance", "Atypical Appearance"]
    # Submission Keys: "negative", "typical", "indeterminate", "atypical"
    label_map = ["negative", "typical", "indeterminate", "atypical"]

    for sid, preds in zip(study_ids, study_preds):
        prediction_parts = []
        for i, label in enumerate(label_map):
            score = preds[i]
            # Format: "class_id confidence 0 0 1 1"
            prediction_parts.append(f"{label} {score} 0 0 1 1")

        prediction_string = " ".join(prediction_parts)
        rows.append({"id": sid, "PredictionString": prediction_string})

    # 2. Process Image Predictions
    for iid, preds in zip(image_ids, image_preds):
        boxes = preds["boxes"]
        scores = preds["scores"]

        if len(boxes) == 0:
            # No findings
            # Format: "none 1 0 0 1 1"
            prediction_string = "none 1 0 0 1 1"
        else:
            # Opacity findings
            box_strings = []
            for box, score in zip(boxes, scores):
                # Format: "opacity confidence xmin ymin xmax ymax"
                b_str = f"opacity {score} {box[0]} {box[1]} {box[2]} {box[3]}"
                box_strings.append(b_str)
            prediction_string = " ".join(box_strings)

        rows.append({"id": iid, "PredictionString": prediction_string})

    # Create DataFrame and save
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)

    return df
