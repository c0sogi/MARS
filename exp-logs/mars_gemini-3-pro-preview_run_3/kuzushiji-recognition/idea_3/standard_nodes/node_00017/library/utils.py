import numpy as np
import torch
import torch.nn.functional as F
from library.config import Config

# ==========================================
# 1. Gaussian Utilities for Heatmap Generation
# ==========================================


def gaussian_radius(det_size, min_overlap=0.7):
    """
    Calculates the radius for the Gaussian kernel based on the object size
    and a minimum IoU overlap constraint.
    Derived from CornerNet/CenterNet.

    Args:
        det_size (tuple): (height, width) of the bounding box.
        min_overlap (float): Minimum IoU overlap.

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


def draw_umich_gaussian(heatmap, center, radius, k=1):
    """
    Draws a 2D Gaussian on the heatmap in-place.

    Args:
        heatmap (np.ndarray): The heatmap array (H, W).
        center (tuple): (x, y) coordinates of the center.
        radius (float): Radius of the Gaussian.
        k (float): Scaling factor (amplitude).

    Returns:
        np.ndarray: The modified heatmap.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0], heatmap.shape[1]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[
        radius - top : radius + bottom, radius - left : radius + right
    ]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap


def gaussian2D(shape, sigma=1):
    """
    Generates a 2D Gaussian kernel.
    """
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


# ==========================================
# 2. Decoding and NMS
# ==========================================


def _nms_heatmap(heat, kernel=3):
    """
    Applies Max Pooling to find local maxima in the heatmap.
    """
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _gather_feat(feat, ind, mask=None):
    """
    Gathers values from a feature map at specific indices.
    """
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    """
    Transposes feature map and gathers values.
    """
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def ctdet_decode(heat, wh, reg, K=100):
    """
    Decodes the output of the CenterNet detector into bounding boxes.

    Args:
        heat (torch.Tensor): Heatmap output (B, C, H, W).
        wh (torch.Tensor): Width/Height output (B, 2, H, W).
        reg (torch.Tensor): Offset regression output (B, 2, H, W).
        K (int): Top K detections to keep.

    Returns:
        torch.Tensor: Detections of shape (B, K, 6) -> [x1, y1, x2, y2, score, class]
                      Coordinates are in the feature map scale.
    """
    batch, cat, height, width = heat.size()

    # 1. Find local maxima
    heat = _nms_heatmap(heat)

    # 2. Get top K scores
    scores, inds, clses, ys, xs = _topk(heat, K=K)

    # 3. Get offset and size at peak locations
    if reg is not None:
        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5

    wh = _transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 2)

    # 4. Convert center + wh to bounding box
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    # xs, ys are centers. wh is width, height
    # Box format: x1, y1, x2, y2
    bboxes = torch.cat(
        [
            xs - wh[..., 0:1] / 2,
            ys - wh[..., 1:2] / 2,
            xs + wh[..., 0:1] / 2,
            ys + wh[..., 1:2] / 2,
        ],
        dim=2,
    )

    detections = torch.cat([bboxes, scores, clses], dim=2)
    return detections


def _topk(scores, K=40):
    """
    Extracts top K scores and their indices from the heatmap.
    """
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds // width).float()
    topk_xs = (topk_inds % width).float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind // K).float()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def nms(boxes, scores, overlap_thresh=0.5):
    """
    Standard Non-Maximum Suppression (NMS).

    Args:
        boxes (np.ndarray): Bounding boxes (N, 4) [x1, y1, x2, y2].
        scores (np.ndarray): Confidence scores (N,).
        overlap_thresh (float): IoU threshold.

    Returns:
        list: Indices of boxes to keep.
    """
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(
            idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0]))
        )

    return pick


# ==========================================
# 3. Metric Calculation
# ==========================================


def calc_modified_f1(predictions, ground_truths):
    """
    Calculates the modified F1 score based on the competition metric.

    Metric Definition:
    - True Positive: Predicted center point is within GT box AND label matches.
    - Ground truth boxes are {label X Y Width Height}.
    - Predictions are {label X Y}.

    Args:
        predictions (list of dict): List where each item is a dict of predictions for an image.
            Format: {'image_id': str, 'preds': [(label, x, y), ...]}
        ground_truths (list of dict): List where each item is a dict of GT for an image.
            Format: {'image_id': str, 'gt': [(label, x, y, w, h), ...]}

    Returns:
        dict: {'f1': float, 'precision': float, 'recall': float}
    """
    tp_total = 0
    fp_total = 0
    fn_total = 0

    # Create lookup for GT
    gt_map = {item["image_id"]: item["gt"] for item in ground_truths}

    for pred_item in predictions:
        image_id = pred_item["image_id"]
        preds = pred_item["preds"]  # List of (label, x, y)

        if image_id not in gt_map:
            # If no GT for this image, all preds are FP
            fp_total += len(preds)
            continue

        gts = gt_map[image_id]  # List of (label, x, y, w, h)

        # Track which GTs have been matched
        gt_matched = [False] * len(gts)

        # Iterate through predictions
        for p_label, px, py in preds:
            match_found = False

            # Check against all unmatched GTs
            for i, (g_label, gx, gy, gw, gh) in enumerate(gts):
                if gt_matched[i]:
                    continue

                # Check label match
                if p_label != g_label:
                    continue

                # Check point in box
                if (gx <= px <= gx + gw) and (gy <= py <= gy + gh):
                    gt_matched[i] = True
                    match_found = True
                    break

            if match_found:
                tp_total += 1
            else:
                fp_total += 1

        # Count False Negatives (unmatched GTs)
        fn_total += sum(1 for m in gt_matched if not m)

    # Handle images in GT but not in Preds (all FN)
    pred_ids = set(p["image_id"] for p in predictions)
    for img_id, gt_list in gt_map.items():
        if img_id not in pred_ids:
            fn_total += len(gt_list)

    epsilon = 1e-7
    precision = tp_total / (tp_total + fp_total + epsilon)
    recall = tp_total / (tp_total + fn_total + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
    }
