import cv2
import numpy as np
from library.config import Config


def get_affine_transform(img_shape, output_size, inverse=False):
    """
    Generates an affine transformation matrix to resize an image to output_size
    while preserving aspect ratio (letterboxing) and centering the image.

    Args:
        img_shape (tuple): (height, width) of the source image.
        output_size (tuple or int): (height, width) or side length of the target image.
        inverse (bool): If True, returns the inverse transform (Target -> Source).

    Returns:
        np.ndarray: 2x3 affine transformation matrix.
    """
    h, w = img_shape
    if isinstance(output_size, int):
        dst_h, dst_w = output_size, output_size
    else:
        dst_h, dst_w = output_size

    # Calculate scaling factor to fit within destination while preserving aspect ratio
    scale = min(dst_w / w, dst_h / h)

    # Calculate new dimensions
    nw = int(w * scale)
    nh = int(h * scale)

    # Calculate translation to center the resized image
    tx = (dst_w - nw) / 2
    ty = (dst_h - nh) / 2

    # Construct the transformation matrix
    # Format: [[scale, 0, tx], [0, scale, ty]]
    trans = np.array([[scale, 0, tx], [0, scale, ty]], dtype=np.float32)

    if inverse:
        # Calculate inverse affine transform
        trans = cv2.invertAffineTransform(trans)

    return trans


def affine_transform(pt, t):
    """
    Applies an affine transformation to a single 2D point.

    Args:
        pt (tuple or list or np.ndarray): (x, y) coordinates.
        t (np.ndarray): 2x3 affine transformation matrix.

    Returns:
        np.ndarray: Transformed (x, y) coordinates.
    """
    new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
    res = np.dot(t, new_pt)
    return res[:2]


def calc_f1_score(predictions, ground_truths):
    """
    Calculates the modified F1 score for the competition.

    Metric Definition:
    To score a true positive, you must provide center point coordinates that are
    within the ground truth bounding box and a matching label.

    Args:
        predictions (list): List of predictions for each image.
                            Each element is a list of dicts:
                            {'point': (x, y), 'label': class_id, 'score': float}
        ground_truths (list): List of ground truths for each image.
                              Each element is a list of dicts:
                              {'box': (x, y, w, h), 'label': class_id}

    Returns:
        float: The global F1 score.
    """
    tp = 0
    fp = 0
    fn = 0

    for preds, gts in zip(predictions, ground_truths):
        # Sort predictions by score descending to prioritize high-confidence detections
        preds_sorted = sorted(preds, key=lambda x: x.get("score", 0), reverse=True)

        # Keep track of matched GTs to avoid double counting (one GT can only match one Pred)
        gt_matched = [False] * len(gts)

        for p in preds_sorted:
            p_x, p_y = p["point"]
            p_label = p["label"]

            match_found = False

            # Iterate through GTs to find a match
            for i, gt in enumerate(gts):
                if gt_matched[i]:
                    continue

                gt_label = gt["label"]
                gt_x, gt_y, gt_w, gt_h = gt["box"]

                # Check label match
                if p_label != gt_label:
                    continue

                # Check if point is inside box
                # Box format is x, y, w, h
                if (gt_x <= p_x <= gt_x + gt_w) and (gt_y <= p_y <= gt_y + gt_h):
                    gt_matched[i] = True
                    match_found = True
                    break

            if match_found:
                tp += 1
            else:
                fp += 1

        # Count False Negatives (unmatched GTs)
        fn += sum(1 for m in gt_matched if not m)

    # Calculate Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return f1
