import math
import numpy as np
import cv2
from library.config import Config


def gaussian_radius(det_size, min_overlap=0.7):
    """
    Calculates the gaussian radius for a bounding box based on the object size.
    Derived from CornerNet/CenterNet logic to ensure the generated heatmap
    peaks overlap sufficiently with the ground truth.

    Args:
        det_size (tuple): (height, width) of the bounding box.
        min_overlap (float): Minimum overlap required (default 0.7).

    Returns:
        float: The calculated radius.
    """
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)


def gaussian_2d(shape, sigma=1):
    """
    Generates a 2D Gaussian kernel.
    """
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1):
    """
    Draws a Gaussian distribution on the heatmap at the specified center.
    Used for generating ground truth heatmaps for the detector.

    Args:
        heatmap (np.ndarray): The heatmap to draw on (H, W).
        center (tuple): (x, y) coordinates of the peak.
        radius (float): Radius of the Gaussian.
        k (float): Scaling factor (usually 1).

    Returns:
        np.ndarray: The modified heatmap.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[
        radius - top : radius + bottom, radius - left : radius + right
    ]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap


def resize_with_padding(image, target_size=Config.DETECTOR_INPUT_SIZE):
    """
    Resizes an image to fit within target_size x target_size while preserving aspect ratio.
    The shorter side is padded with zeros (black) to make the output square.

    Args:
        image (np.ndarray): Input image (H, W, C).
        target_size (int): Target dimension for the square output.

    Returns:
        tuple: (padded_image, scale, pad_w, pad_h)
            - padded_image: The resized and padded image.
            - scale: The scaling factor applied.
            - pad_w: Total padding applied to width.
            - pad_h: Total padding applied to height.
    """
    h, w = image.shape[:2]

    # Calculate scale to fit the longest side
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Resize
    resized_image = cv2.resize(image, (new_w, new_h))

    # Create square canvas
    canvas = np.zeros((target_size, target_size, image.shape[2]), dtype=image.dtype)

    # Paste resized image at top-left (0,0)
    canvas[:new_h, :new_w, :] = resized_image

    pad_w = target_size - new_w
    pad_h = target_size - new_h

    return canvas, scale, pad_w, pad_h


def get_dir(src_point, rot_rad):
    """Helper for get_affine_transform used in rotation calculation."""
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_3rd_point(a, b):
    """Helper for get_affine_transform to calculate the third point of the triangle."""
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(
    center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0
):
    """
    Generates an affine transformation matrix for cropping/resizing/rotating.
    Standard implementation for CenterNet-style data augmentation.

    Args:
        center (np.ndarray): Center of the crop [x, y].
        scale (float or np.ndarray): Scale factor.
        rot (float): Rotation in degrees.
        output_size (list/tuple): [width, height] of output.
        shift (np.ndarray): Shift factor.
        inv (int): If 1, returns inverse transform.

    Returns:
        np.ndarray: 2x3 Affine transformation matrix.
    """
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(dst, src)
    else:
        trans = cv2.getAffineTransform(src, dst)

    return trans


def affine_transform(pt, t):
    """
    Applies an affine transformation to a 2D point.

    Args:
        pt (np.ndarray/list): Point [x, y].
        t (np.ndarray): 2x3 Transformation matrix.

    Returns:
        np.ndarray: Transformed point [x, y].
    """
    new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def modified_f1_score(df_true, df_pred):
    """
    Calculates the modified F1 Score for the Kuzushiji recognition task.

    Metric Definition:
    - True Positive: Center point within GT bounding box AND matching label.
    - False Positive: Prediction not matching any GT or duplicate match.
    - False Negative: GT not matched by any prediction.

    Args:
        df_true (pd.DataFrame): DataFrame containing 'image_id' and 'labels' (GT).
        df_pred (pd.DataFrame): DataFrame containing 'image_id' and 'labels' (Predictions).

    Returns:
        tuple: (f1_score, precision, recall)
    """
    # Map image_id to label string, handling NaNs
    true_map = dict(zip(df_true["image_id"], df_true["labels"].fillna("")))
    pred_map = dict(zip(df_pred["image_id"], df_pred["labels"].fillna("")))

    tp_total = 0
    fp_total = 0
    fn_total = 0

    all_ids = set(true_map.keys()) | set(pred_map.keys())

    for img_id in all_ids:
        # Parse Ground Truth
        # Format: Code X Y W H ...
        gt_str = true_map.get(img_id, "")
        gts = []
        if gt_str:
            parts = gt_str.split()
            # Each GT is 5 tokens
            count = len(parts) // 5
            for i in range(count):
                try:
                    gts.append(
                        {
                            "code": parts[i * 5],
                            "x": int(parts[i * 5 + 1]),
                            "y": int(parts[i * 5 + 2]),
                            "w": int(parts[i * 5 + 3]),
                            "h": int(parts[i * 5 + 4]),
                            "matched": False,
                        }
                    )
                except ValueError:
                    continue

        # Parse Predictions
        # Format: Code X Y ...
        pred_str = pred_map.get(img_id, "")
        preds = []
        if pred_str:
            parts = pred_str.split()
            # Each Pred is 3 tokens
            count = len(parts) // 3
            for i in range(count):
                try:
                    preds.append(
                        {
                            "code": parts[i * 3],
                            "x": int(parts[i * 3 + 1]),
                            "y": int(parts[i * 3 + 2]),
                        }
                    )
                except ValueError:
                    continue

        # Match Predictions to GT
        # Greedy matching: iterate preds and find first available GT
        for p in preds:
            match_found = False
            for g in gts:
                if g["matched"]:
                    continue

                # Check Label
                if p["code"] != g["code"]:
                    continue

                # Check Geometry: Center point within BBox
                # BBox is (x, y, w, h)
                if (g["x"] <= p["x"] < g["x"] + g["w"]) and (
                    g["y"] <= p["y"] < g["y"] + g["h"]
                ):

                    g["matched"] = True
                    match_found = True
                    break

            if match_found:
                tp_total += 1
            else:
                fp_total += 1

        # Count False Negatives (unmatched GTs)
        fn_total += sum(1 for g in gts if not g["matched"])

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0

    if (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return f1, precision, recall
